import { Buffer } from "node:buffer";

export default {
  async fetch(request) {
    if (request.method === "OPTIONS") {
      return handleOPTIONS();
    }
    const errHandler = (err) => {
      console.error(err);
      return jsonResponse({ error: err.message }, fixCors({ status: err.status ?? 500 }));
    };
    try {
      const auth = request.headers.get("Authorization");
      const apiKey = auth?.split(" ")[1];
      const assert = (success) => {
        if (!success) {
          throw new HttpError("The specified HTTP method is not allowed for the requested resource", 400);
        }
      };
      const { pathname } = new URL(request.url);
      switch (true) {
        case pathname.endsWith("/chat/completions"):
          assert(request.method === "POST");
          return handleCompletions(await request.json(), apiKey)
            .catch(errHandler);
        case pathname.endsWith("/embeddings"):
          assert(request.method === "POST");
          return handleEmbeddings(await request.json(), apiKey)
            .catch(errHandler);
        case pathname.endsWith("/models"):
          assert(request.method === "GET");
          return handleModels(apiKey)
            .catch(errHandler);
        default:
          throw new HttpError("404 Not Found", 404);
      }
    } catch (err) {
      return errHandler(err);
    }
  }
};

class HttpError extends Error {
  constructor(message, status) {
    super(message);
    this.name = this.constructor.name;
    this.status = status;
  }
}

// 統一的 CORS 與 Content-Encoding
const fixCors = ({ headers, status, statusText }) => {
  headers = new Headers(headers);
  headers.set("Access-Control-Allow-Origin", "*");
  headers.set("Content-Encoding", "identity");
  return { headers, status, statusText };
};

// 統一 JSON 回應
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

// CORS OPTIONS
const handleOPTIONS = async () => {
  return new Response(null, {
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "*",
      "Access-Control-Allow-Headers": "*",
      "Content-Encoding": "identity"
    }
  });
};

const BASE_URL = "https://generativelanguage.googleapis.com";
const API_VERSION = "v1beta";
const API_CLIENT = "genai-js/0.21.0";
const makeHeaders = (apiKey, more) => ({
  "x-goog-api-client": API_CLIENT,
  ...(apiKey && { "x-goog-api-key": apiKey }),
  ...more
});

// ===== models
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

// ===== embeddings
const DEFAULT_EMBEDDINGS_MODEL = "text-embedding-004";
async function handleEmbeddings(req, apiKey) {
  if (typeof req.model !== "string") {
    throw new HttpError("model is not specified", 400);
  }
  let model;
  if (req.model.startsWith("models/")) {
    model = req.model;
  } else {
    if (!req.model.startsWith("gemini-")) {
      req.model = DEFAULT_EMBEDDINGS_MODEL;
    }
    model = "models/" + req.model;
  }
  if (!Array.isArray(req.input)) {
    req.input = [req.input];
  }
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

// ===== chat completions
const DEFAULT_MODEL = "gemini-2.5-flash";
async function handleCompletions(req, apiKey) {
  let model = DEFAULT_MODEL;
  switch (true) {
    case typeof req.model !== "string":
      break;
    case req.model.startsWith("models/"):
      model = req.model.substring(7);
      break;
    case req.model.startsWith("gemini-"):
    case req.model.startsWith("gemma-"):
    case req.model.startsWith("learnlm-"):
      model = req.model;
  }
  let body = await transformRequest(req);
  const extra = req.extra_body?.google;
  if (extra) {
    if (extra.safety_settings) body.safetySettings = extra.safety_settings;
    if (extra.cached_content) body.cachedContent = extra.cached_content;
    if (extra.thinking_config) body.generationConfig.thinkingConfig = extra.thinking_config;
  }
  switch (true) {
    case model.endsWith(":search"):
      model = model.substring(0, model.length - 7);
    case req.model.endsWith("-search-preview"):
      body.tools = body.tools || [];
      body.tools.push({ googleSearch: {} });
  }
  const TASK = req.stream ? "streamGenerateContent" : "generateContent";
  let url = `${BASE_URL}/${API_VERSION}/models/${model}:${TASK}`;
  if (req.stream) { url += "?alt=sse"; }
  const response = await fetch(url, {
    method: "POST",
    headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
    body: JSON.stringify(body),
  });

  if (response.ok) {
    let id = "chatcmpl-" + generateId();
    if (req.stream) {
      // 流式 SSE 自己加 header
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
    return jsonResponse(errorText, fixCors(response));
  }
}

// --------------- 下面維持原本各種轉換 function（原樣即可） ----------------
// generateId, transformRequest, processCompletionsResponse, 等等
// 如果你有現成內容就直接複製，如果要完整內容請告訴我！

// ------------------------------------------------------

function generateId() {
  const characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  const randomChar = () => characters[Math.floor(Math.random() * characters.length)];
  return Array.from({ length: 29 }, randomChar).join("");
}

// ...以下原本轉換 function 可保留...

// ------------------------------------------------------

// 產生唯一ID（已經有在上面）

// Gemini/OAI API 轉換與回應格式化
async function transformRequest(req) {
  // 你可以直接複製之前版本的 transformRequest，或從 https://github.com/PublicAffairs/openai-gemini 取出
  // 示範版（請依你需求調整）：
  return {
    contents: [{
      role: "user",
      parts: [{ text: req.messages?.map(m => m.content).join("\n") }]
    }],
    generationConfig: {
      maxOutputTokens: req.max_tokens || 2048,
      temperature: req.temperature || 0.9
    },
    // ...其他需要的欄位
  };
}

// 處理 Gemini Completion 回應，轉換成 OAI 格式
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
