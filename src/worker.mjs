import { Buffer } from "node:buffer";

// ===== 幫助圖片轉換的輔助方法 =====
const parseImg = async (url) => {
  let mimeType, data;
  if (url.startsWith("http://") || url.startsWith("https://")) {
    // 線上圖片自動下載並 base64
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to download image: ${url} - ${response.statusText}`);
    }
    mimeType = response.headers.get("content-type") || "image/jpeg";
    data = Buffer.from(await response.arrayBuffer()).toString("base64");
  } else {
    // data url or base64
    const match = url.match(/^data:(?<mimeType>.*?)(;base64)?,(?<data>.*)$/);
    if (!match) throw new Error("Invalid image data: " + url);
    ({ mimeType, data } = match.groups);
  }
  return { inlineData: { mimeType, data } };
};

// ====== 主 export ======
export default {
  async fetch(request) {
    if (request.method === "OPTIONS") return handleOPTIONS();

    try {
      const auth = request.headers.get("Authorization");
      const apiKey = auth?.split(" ")[1];
      const { pathname } = new URL(request.url);

      switch (true) {
        case pathname.endsWith("/chat/completions"):
          if (request.method !== "POST") return notAllowed();
          return await handleCompletions(await request.json(), apiKey);
        case pathname.endsWith("/embeddings"):
          if (request.method !== "POST") return notAllowed();
          return await handleEmbeddings(await request.json(), apiKey);
        case pathname.endsWith("/models"):
          if (request.method !== "GET") return notAllowed();
          return await handleModels(apiKey);
        default:
          return notFound();
      }
    } catch (err) {
      console.error("[WORKER ERROR]", err);
      return jsonResponse({ error: err.message || String(err) }, fixCors({ status: 500 }));
    }
  }
};

// ============= 各種 handler 實作 =============

const BASE_URL = "https://generativelanguage.googleapis.com";
const API_VERSION = "v1beta";
const API_CLIENT = "genai-js/0.21.0";
const DEFAULT_MODEL = "gemini-2.5-flash";
const DEFAULT_EMBEDDINGS_MODEL = "text-embedding-004";

// 統一 header/cors
function fixCors({ headers, status, statusText }) {
  headers = new Headers(headers);
  headers.set("Access-Control-Allow-Origin", "*");
  headers.set("Content-Encoding", "identity");
  return { headers, status, statusText };
}
function jsonResponse(body, base = {}) {
  return new Response(
    typeof body === "string" ? body : JSON.stringify(body),
    {
      ...base,
      headers: {
        ...(base.headers || {}),
        "Content-Type": "application/json",
        "Content-Encoding": "identity"
      }
    }
  );
}
const handleOPTIONS = async () => new Response(null, {
  headers: {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "*",
    "Access-Control-Allow-Headers": "*",
    "Content-Encoding": "identity"
  }
});
const notFound = () => jsonResponse({ error: "404 Not Found" }, fixCors({ status: 404 }));
const notAllowed = () => jsonResponse({ error: "405 Method Not Allowed" }, fixCors({ status: 405 }));

const makeHeaders = (apiKey, more) => ({
  "x-goog-api-client": API_CLIENT,
  ...(apiKey && { "x-goog-api-key": apiKey }),
  ...more
});

async function handleModels(apiKey) {
  const response = await fetch(`${BASE_URL}/${API_VERSION}/models`, {
    headers: makeHeaders(apiKey),
  });
  let { body } = response;
  if (response.ok) {
    const { models } = JSON.parse(await response.text());
    body = JSON.stringify({
      object: "list",
      data: models.map(({ name }) => ({
        id: name.replace("models/", ""),
        object: "model",
        created: 0,
        owned_by: "",
      })),
    });
  }
  return jsonResponse(body, fixCors(response));
}

async function handleEmbeddings(req, apiKey) {
  if (typeof req.model !== "string") throw new Error("model is not specified");
  let model = req.model.startsWith("models/") ? req.model : "models/" + (req.model.startsWith("gemini-") ? req.model : DEFAULT_EMBEDDINGS_MODEL);
  if (!Array.isArray(req.input)) req.input = [req.input];
  const response = await fetch(`${BASE_URL}/${API_VERSION}/${model}:batchEmbedContents`, {
    method: "POST",
    headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
    body: JSON.stringify({
      "requests": req.input.map(text => ({
        model,
        content: { parts: { text } },
        outputDimensionality: req.dimensions,
      }))
    })
  });
  let { body } = response;
  if (response.ok) {
    const { embeddings } = JSON.parse(await response.text());
    body = JSON.stringify({
      object: "list",
      data: embeddings.map(({ values }, index) => ({
        object: "embedding",
        index,
        embedding: values,
      })),
      model: req.model,
    });
  }
  return jsonResponse(body, fixCors(response));
}

async function handleCompletions(req, apiKey) {
  let model = DEFAULT_MODEL;
  if (typeof req.model === "string") {
    if (req.model.startsWith("models/")) model = req.model.substring(7);
    else model = req.model;
  }
  let body = await transformRequest(req);
  // 支援 Google Search 工具
  if (model.endsWith(":search")) {
    model = model.substring(0, model.length - 7);
    body.tools = body.tools || [];
    body.tools.push({ googleSearch: {} });
  }
  const TASK = req.stream ? "streamGenerateContent" : "generateContent";
  let url = `${BASE_URL}/${API_VERSION}/models/${model}:${TASK}`;
  if (req.stream) url += "?alt=sse";
  const response = await fetch(url, {
    method: "POST",
    headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
    body: JSON.stringify(body),
  });

  if (response.ok) {
    let id = "chatcmpl-" + generateId();
    if (req.stream) {
      // 流式 SSE
      let sseHeaders = fixCors(response);
      sseHeaders.headers.set("Content-Type", "text/event-stream");
      return new Response(response.body, sseHeaders);
    } else {
      let resText = await response.text();
      try {
        let resObj = JSON.parse(resText);
        if (!resObj.candidates) throw new Error("Invalid completion object");
        resText = processCompletionsResponse(resObj, model, id);
      } catch (err) {
        console.error("Error parsing response:", err);
        return jsonResponse(resText, fixCors(response));
      }
      return jsonResponse(resText, fixCors(response));
    }
  } else {
    let errorText = await response.text();
    console.error("[Gemini API ERROR]", errorText);
    return jsonResponse(errorText, fixCors(response));
  }
}

// ===== 轉換 messages 支援 vision =====
async function transformRequest(req) {
  // messages: [{role, content: [{type: "text"|"image_url", text|image_url:{url}}]}]
  const parts = [];
  for (const msg of req.messages || []) {
    if (msg.content && Array.isArray(msg.content)) {
      for (const item of msg.content) {
        if (item.type === "image_url" && item.image_url?.url) {
          parts.push(await parseImg(item.image_url.url));
        } else if (item.type === "text" && item.text) {
          parts.push({ text: item.text });
        }
      }
    } else if (typeof msg.content === "string") {
      parts.push({ text: msg.content });
    }
  }
  if (parts.length === 0) parts.push({ text: "" }); // 至少有一個 text
  return {
    contents: [{ role: "user", parts }],
    generationConfig: {
      maxOutputTokens: req.max_tokens || 2048,
      temperature: req.temperature || 0.9
    }
  };
}

function processCompletionsResponse(data, model, id) {
  return JSON.stringify({
    id,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: data.candidates.map((c, idx) => ({
      index: idx,
      message: { role: "assistant", content: c.content?.parts?.map(p => p.text).join("") || "" },
      finish_reason: c.finishReason || "stop"
    })),
    usage: data.usageMetadata ? {
      completion_tokens: data.usageMetadata.candidatesTokenCount,
      prompt_tokens: data.usageMetadata.promptTokenCount,
      total_tokens: data.usageMetadata.totalTokenCount
    } : undefined
  });
}

function generateId() {
  const characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  const randomChar = () => characters[Math.floor(Math.random() * characters.length)];
  return Array.from({ length: 29 }, randomChar).join("");
}
