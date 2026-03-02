"""
AI Memory Gateway — 带记忆系统的 LLM 转发网关
=============================================
让你的 AI 拥有长期记忆。

工作原理：
1. 接收客户端（Kelivo / ChatBox / 任何 OpenAI 兼容客户端）的消息
2. 自动搜索数据库中的相关记忆，注入 system prompt
3. 转发给 LLM API（支持 OpenRouter / OpenAI / 任何兼容接口）
4. 后台自动存储对话 + 用 AI 提取新记忆

环境变量 MEMORY_ENABLED=false 时退化为纯转发网关（第一阶段）。
"""

import os
import json
import uuid
import asyncio
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse

from database import init_tables, close_pool, save_message, search_memories, save_memory, get_all_memories_count, get_recent_memories, get_all_memories, get_pool, get_all_memories_detail, update_memory, delete_memory, delete_memories_batch
from memory_extractor import extract_memories, score_memories

# ============================================================
# 配置项 —— 全部从环境变量读取，部署时在云平台面板里设置
# ============================================================

# 你的 API Key（OpenRouter / OpenAI / 其他兼容服务）
API_KEY = os.getenv("API_KEY", "")

# API 地址（改这个就能切换不同的 LLM 服务商）
# OpenRouter: https://openrouter.ai/api/v1/chat/completions
# OpenAI:     https://api.openai.com/v1/chat/completions
# 本地 Ollama: http://localhost:11434/v1/chat/completions
API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")

# 默认模型（如果客户端没指定就用这个）
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "anthropic/claude-sonnet-4")

# 网关端口
PORT = int(os.getenv("PORT", "8080"))

# 记忆系统开关（数据库出问题时可以临时关掉）
MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "false").lower() == "true"

# 每次注入的最大记忆条数
MAX_MEMORIES_INJECT = int(os.getenv("MAX_MEMORIES_INJECT", "15"))

# 记忆提取间隔（0 = 禁用自动提取，1 = 每轮提取，N = 每 N 轮提取一次）
MEMORY_EXTRACT_INTERVAL = int(os.getenv("MEMORY_EXTRACT_INTERVAL", "1"))

# 轮次计数器
_round_counter = 0

# 额外的请求头（有些 API 需要，比如 OpenRouter 需要 Referer）
EXTRA_REFERER = os.getenv("EXTRA_REFERER", "https://ai-memory-gateway.local")
EXTRA_TITLE = os.getenv("EXTRA_TITLE", "AI Memory Gateway")


# ============================================================
# 人设加载
# ============================================================

def load_system_prompt():
    """从 system_prompt.txt 文件读取人设内容"""
    prompt_path = os.path.join(os.path.dirname(__file__), "system_prompt.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                return content
    except FileNotFoundError:
        pass
    print("ℹ️  未找到 system_prompt.txt 或文件为空，将不注入 system prompt")
    return ""


SYSTEM_PROMPT = load_system_prompt()
if SYSTEM_PROMPT:
    print(f"✅ 人设已加载，长度：{len(SYSTEM_PROMPT)} 字符")
else:
    print("ℹ️  无人设，纯转发模式")


# ============================================================
# 应用生命周期管理
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动时初始化数据库，关闭时断开连接"""
    if MEMORY_ENABLED:
        try:
            await init_tables()
            count = await get_all_memories_count()
            print(f"✅ 记忆系统已启动，当前记忆数量：{count}")
        except Exception as e:
            print(f"⚠️  数据库初始化失败: {e}")
            print("⚠️  记忆系统将不可用，但网关仍可正常转发")
    else:
        print("ℹ️  记忆系统已关闭（设置 MEMORY_ENABLED=true 开启）")
    
    yield
    
    if MEMORY_ENABLED:
        await close_pool()


app = FastAPI(title="AI Memory Gateway", version="2.0.0", lifespan=lifespan)


# ============================================================
# 记忆注入
# ============================================================

async def build_system_prompt_with_memories(user_message: str) -> str:
    """
    构建带记忆的 system prompt
    1. 用用户消息搜索相关记忆
    2. 格式化成文本拼接到人设后面
    """
    if not MEMORY_ENABLED:
        return SYSTEM_PROMPT
    
    try:
        memories = await search_memories(user_message, limit=MAX_MEMORIES_INJECT)
        
        if not memories:
            return SYSTEM_PROMPT
        
        # 格式化记忆文本（带日期，帮助模型判断新旧）
        memory_lines = []
        for mem in memories:
            date_str = ""
            if mem.get("created_at"):
                try:
                    date_str = f"[{str(mem['created_at'])[:10]}] "
                except:
                    pass
            memory_lines.append(f"- {date_str}{mem['content']}")
        memory_text = "\n".join(memory_lines)
        
        enhanced_prompt = f"""{SYSTEM_PROMPT}

【从过往对话中检索到的相关记忆】
{memory_text}

# 记忆应用
- 像朋友般自然运用这些记忆，不刻意展示
- 仅在相关话题出现时引用，避免主动提及
- 对重要信息（如健康、日期、约定）保持一致性
- 新信息与记忆冲突时，以新信息为准
- 模糊记忆可表达不确定性："记得你似乎说过..."

# 交流方式
- 自然引用："记得你说过..."或"上次我们聊到..."
- 避免机械式表达如"根据我的记忆..."或"检索到的信息显示..."
- 共同经历可温情回忆："上次那个事挺好玩的"

记忆是丰富对话的工具，而非对话焦点。"""
        
        print(f"📚 注入了 {len(memories)} 条相关记忆")
        return enhanced_prompt
        
    except Exception as e:
        print(f"⚠️  记忆检索失败: {e}，使用纯人设")
        return SYSTEM_PROMPT


# ============================================================
# 后台记忆处理
# ============================================================

async def process_memories_background(session_id: str, user_msg: str, assistant_msg: str, model: str, context_messages: list = None):
    """
    后台异步：存储对话 + 提取记忆（不阻塞主流程）
    
    记忆提取受 MEMORY_EXTRACT_INTERVAL 控制：
    - 0: 禁用自动提取
    - 1: 每轮提取（默认）
    - N: 每 N 轮提取一次
    对话记录始终保存，不受间隔影响。
    
    context_messages: 客户端发来的原始对话上下文（不含system prompt），
                      用于让提取模型从完整上下文中提取记忆。
    """
    global _round_counter
    
    try:
        # 1. 存储对话记录（始终执行，只存最新一轮避免重复）
        await save_message(session_id, "user", user_msg, model)
        await save_message(session_id, "assistant", assistant_msg, model)
        
        # 2. 检查是否需要提取记忆
        if MEMORY_EXTRACT_INTERVAL == 0:
            print(f"⏭️  记忆自动提取已禁用，跳过")
            return
        
        _round_counter += 1
        
        if MEMORY_EXTRACT_INTERVAL > 1 and (_round_counter % MEMORY_EXTRACT_INTERVAL != 0):
            print(f"⏭️  轮次 {_round_counter}，跳过记忆提取（每 {MEMORY_EXTRACT_INTERVAL} 轮提取一次）")
            return
        
        if MEMORY_EXTRACT_INTERVAL > 1:
            print(f"📝 轮次 {_round_counter}，执行记忆提取")
        
        # 3. 获取已有记忆，传给提取模型做对比去重
        existing = await get_recent_memories(limit=80)
        existing_contents = [r["content"] for r in existing]
        
        # 4. 构建用于提取的消息列表
        #    截取最近 MEMORY_EXTRACT_INTERVAL 轮对话（每轮=user+assistant共2条）
        #    而非发送完整上下文，省token
        if context_messages:
            # 截取最近N轮（interval×2条），加上最新的assistant回复
            tail_count = MEMORY_EXTRACT_INTERVAL * 2
            recent_msgs = list(context_messages)[-tail_count:] if len(context_messages) > tail_count else list(context_messages)
            messages_for_extraction = recent_msgs + [
                {"role": "assistant", "content": assistant_msg}
            ]
            print(f"📝 截取最近 {MEMORY_EXTRACT_INTERVAL} 轮对话提取记忆（{len(messages_for_extraction)} 条消息）")
        else:
            messages_for_extraction = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
        
        new_memories = await extract_memories(messages_for_extraction, existing_memories=existing_contents)
        
        # 过滤垃圾记忆（不靠模型自觉，硬过滤）
        META_BLACKLIST = [
            "记忆库", "记忆系统", "检索", "没有被记录", "没有被提取",
            "记忆遗漏", "尚未被记录", "写入不完整", "检索功能",
            "系统没有返回", "关键词匹配", "语义匹配", "语义检索",
            "阈值", "数据库", "seed", "导入", "部署",
            "bug", "debug", "端口", "网关",
        ]
        
        filtered_memories = []
        for mem in new_memories:
            content = mem["content"]
            if any(kw in content for kw in META_BLACKLIST):
                print(f"🚫 过滤掉meta记忆: {content[:60]}...")
                continue
            filtered_memories.append(mem)
        
        for mem in filtered_memories:
            await save_memory(
                content=mem["content"],
                importance=mem["importance"],
                source_session=session_id,
            )
        
        if filtered_memories:
            total = await get_all_memories_count()
            print(f"💾 已保存 {len(filtered_memories)} 条新记忆（过滤了 {len(new_memories) - len(filtered_memories)} 条），总计 {total} 条")
            
    except Exception as e:
        print(f"⚠️  后台记忆处理失败: {e}")


# ============================================================
# API 接口
# ============================================================

@app.get("/")
async def health_check():
    """健康检查"""
    memory_count = 0
    if MEMORY_ENABLED:
        try:
            memory_count = await get_all_memories_count()
        except:
            pass
    
    return {
        "status": "running",
        "gateway": "AI Memory Gateway v2.0",
        "system_prompt_loaded": len(SYSTEM_PROMPT) > 0,
        "system_prompt_length": len(SYSTEM_PROMPT),
        "memory_enabled": MEMORY_ENABLED,
        "memory_count": memory_count,
        "memory_extract_interval": MEMORY_EXTRACT_INTERVAL,
    }


@app.get("/v1/models")
async def list_models():
    """模型列表（让客户端不报错）"""
    return {
        "object": "list",
        "data": [
            {
                "id": DEFAULT_MODEL,
                "object": "model",
                "created": 1700000000,
                "owned_by": "ai-memory-gateway",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """核心转发接口"""
    if not API_KEY:
        return JSONResponse(
            status_code=500,
            content={"error": "API_KEY 未设置，请在环境变量中配置"},
        )
    
    body = await request.json()
    messages = body.get("messages", [])
    
    # ---------- 提取用户最新消息 ----------
    user_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                user_message = content
            elif isinstance(content, list):
                user_message = " ".join(
                    item.get("text", "") for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
            break
    
    # ---------- 构建 system prompt ----------
    # 先保存原始对话消息（不含 system prompt），用于记忆提取
    original_messages = [msg for msg in messages if msg.get("role") != "system"]
    
    if SYSTEM_PROMPT or (MEMORY_ENABLED and user_message):
        if MEMORY_ENABLED and user_message:
            enhanced_prompt = await build_system_prompt_with_memories(user_message)
        else:
            enhanced_prompt = SYSTEM_PROMPT
        
        if enhanced_prompt:
            has_system = any(msg.get("role") == "system" for msg in messages)
            if has_system:
                for i, msg in enumerate(messages):
                    if msg.get("role") == "system":
                        messages[i]["content"] = enhanced_prompt + "\n\n" + msg["content"]
                        break
            else:
                messages.insert(0, {"role": "system", "content": enhanced_prompt})
    
    body["messages"] = messages
    
    # ---------- 模型处理 ----------
    model = body.get("model", DEFAULT_MODEL)
    if not model:
        model = DEFAULT_MODEL
    body["model"] = model
    
    # ---------- 生成 session ID ----------
    session_id = str(uuid.uuid4())[:8]
    
    # ---------- 转发请求 ----------
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    # OpenRouter 需要的额外头
    if "openrouter" in API_BASE_URL:
        headers["HTTP-Referer"] = EXTRA_REFERER
        headers["X-Title"] = EXTRA_TITLE
    
    is_stream = body.get("stream", False)
    
    if is_stream:
        return StreamingResponse(
            stream_and_capture(headers, body, session_id, user_message, model, original_messages),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    else:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(API_BASE_URL, headers=headers, json=body)
            
            if response.status_code == 200:
                resp_data = response.json()
                assistant_msg = ""
                try:
                    assistant_msg = resp_data["choices"][0]["message"]["content"]
                except (KeyError, IndexError):
                    pass
                
                if MEMORY_ENABLED and user_message and assistant_msg:
                    asyncio.create_task(
                        process_memories_background(session_id, user_message, assistant_msg, model, context_messages=original_messages)
                    )
                
                return JSONResponse(status_code=200, content=resp_data)
            else:
                return JSONResponse(status_code=response.status_code, content=response.json())


async def stream_and_capture(headers: dict, body: dict, session_id: str, user_message: str, model: str, original_messages: list = None):
    """流式响应 + 捕获完整回复"""
    full_response = []
    
    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream("POST", API_BASE_URL, headers=headers, json=body) as response:
            async for line in response.aiter_lines():
                if line:
                    yield line + "\n"
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            data = json.loads(line[6:])
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_response.append(content)
                        except (json.JSONDecodeError, KeyError, IndexError):
                            pass
    
    assistant_msg = "".join(full_response)
    if MEMORY_ENABLED and user_message and assistant_msg:
        asyncio.create_task(
            process_memories_background(session_id, user_message, assistant_msg, model, context_messages=original_messages)
        )


# ============================================================
# 记忆管理接口
# ============================================================


@app.get("/import/seed-memories")
async def import_seed_memories():
    """一次性导入预置记忆（从 seed_memories.py）"""
    try:
        from seed_memories import run_seed_import
        result = await run_seed_import()
        return result
    except ImportError:
        return {"error": "未找到 seed_memories.py，请参考 seed_memories_example.py 创建"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/export/memories")
async def export_memories():
    """
    导出所有记忆为 JSON（用于备份或迁移）
    浏览器访问这个地址就会返回所有记忆数据
    """
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用（设置 MEMORY_ENABLED=true 开启）"}
    
    try:
        memories = await get_all_memories()
        # 把 datetime 转成字符串
        for mem in memories:
            if mem.get("created_at"):
                mem["created_at"] = str(mem["created_at"])
        
        return {
            "total": len(memories),
            "exported_at": str(__import__("datetime").datetime.now()),
            "memories": memories,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/import/memories", response_class=HTMLResponse)
async def import_memories_page():
    """导入记忆的网页界面"""
    if not MEMORY_ENABLED:
        return HTMLResponse("<h3>记忆系统未启用（设置 MEMORY_ENABLED=true 开启）</h3>")
    
    return HTMLResponse("""
<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>导入记忆</title>
<style>
    body { font-family: sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; }
    textarea { width: 100%%; height: 200px; font-size: 14px; margin: 10px 0; }
    button { padding: 10px 20px; font-size: 16px; cursor: pointer; background: #4CAF50; color: white; border: none; border-radius: 4px; margin-right: 8px; }
    button:hover { background: #45a049; }
    input[type="file"] { margin: 10px 0; font-size: 14px; }
    #result { margin-top: 15px; padding: 10px; white-space: pre-wrap; }
    .ok { background: #e8f5e9; } .err { background: #ffebee; } .info { background: #e3f2fd; }
    .tabs { display: flex; gap: 0; margin-bottom: 20px; border-bottom: 2px solid #eee; }
    .tab { padding: 10px 20px; cursor: pointer; border-bottom: 2px solid transparent; margin-bottom: -2px; color: #666; }
    .tab.active { border-bottom-color: #4CAF50; color: #333; font-weight: bold; }
    .panel { display: none; } .panel.active { display: block; }
    .hint { color: #888; font-size: 13px; margin: 5px 0; }
    label { cursor: pointer; }
    .preview { background: #f5f5f5; border: 1px solid #ddd; padding: 10px; margin: 10px 0; max-height: 200px; overflow-y: auto; font-size: 13px; }
    .preview-item { padding: 3px 0; border-bottom: 1px solid #eee; }
    .nav { margin-bottom: 15px; font-size: 14px; color: #666; }
    .nav a { color: #4CAF50; text-decoration: none; }
</style></head><body>
<h2>📥 导入记忆</h2>
<div class="nav"><a href="/manage/memories">→ 管理已有记忆</a></div>

<div class="tabs">
    <div class="tab active" onclick="switchTab('text')">纯文本导入</div>
    <div class="tab" onclick="switchTab('json')">JSON 备份恢复</div>
</div>

<div id="panel-text" class="panel active">
    <p>上传 <b>.txt 文件</b>（每行一条记忆），或直接在下方输入。</p>
    <p class="hint">示例：一行写一条，比如 "用户的名字叫小花"、"用户喜欢吃火锅"</p>
    <input type="file" id="txtFile" accept=".txt">
    <div style="margin: 15px 0; text-align: center; color: #999;">—— 或者直接输入 ——</div>
    <textarea id="txtInput" placeholder="每行一条记忆，例如：&#10;用户的名字叫小花&#10;用户喜欢吃火锅&#10;用户养了一只狗叫豆豆"></textarea>
    <p><label><input type="checkbox" id="skipScore"> 跳过自动评分（所有记忆默认权重 5，不消耗 API 额度）</label></p>
    <button onclick="doTextImport()">导入</button>
</div>

<div id="panel-json" class="panel">
    <p>上传从 <code>/export/memories</code> 保存的 <b>.json 文件</b>，用于备份恢复或平台迁移。</p>
    <input type="file" id="jsonFile" accept=".json">
    <div style="margin: 15px 0; text-align: center; color: #999;">—— 或者直接粘贴 ——</div>
    <textarea id="jsonInput" placeholder="粘贴导出的 JSON"></textarea>
    <br><button onclick="previewJson()">预览</button>
    <div id="jsonPreview"></div>
</div>

<div id="result"></div>

<script>
function switchTab(name) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    event.target.classList.add('active');
    document.getElementById('panel-' + name).classList.add('active');
    document.getElementById('result').textContent = '';
    document.getElementById('result').className = '';
    document.getElementById('jsonPreview').innerHTML = '';
}

async function doTextImport() {
    const r = document.getElementById('result');
    const file = document.getElementById('txtFile').files[0];
    const text = document.getElementById('txtInput').value.trim();
    const skip = document.getElementById('skipScore').checked;
    
    let content = '';
    if (file) { content = await file.text(); }
    else if (text) { content = text; }
    else { r.className = 'err'; r.textContent = '请先上传文件或输入文本'; return; }
    
    const lines = content.split('\\n').map(l => l.trim()).filter(l => l.length > 0);
    if (lines.length === 0) { r.className = 'err'; r.textContent = '没有找到有效的记忆条目'; return; }
    
    r.className = 'info';
    r.textContent = skip ? '正在导入 ' + lines.length + ' 条记忆...' : '正在为 ' + lines.length + ' 条记忆自动评分，请稍候...';
    
    try {
        const resp = await fetch('/import/text', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({lines: lines, skip_scoring: skip})
        });
        const data = await resp.json();
        if (data.error) { r.className = 'err'; r.textContent = '❌ ' + data.error; }
        else { r.className = 'ok'; r.textContent = '✅ 导入完成！新增 ' + data.imported + ' 条，跳过 ' + data.skipped + ' 条（已存在），总计 ' + data.total + ' 条'; }
    } catch(e) { r.className = 'err'; r.textContent = '❌ 请求失败：' + e.message; }
}

let pendingJsonData = null;

async function previewJson() {
    const r = document.getElementById('result');
    const p = document.getElementById('jsonPreview');
    const file = document.getElementById('jsonFile').files[0];
    const text = document.getElementById('jsonInput').value.trim();
    
    let jsonStr = '';
    if (file) { jsonStr = await file.text(); }
    else if (text) { jsonStr = text; }
    else { r.className = 'err'; r.textContent = '请先上传文件或粘贴 JSON'; return; }
    
    try {
        const parsed = JSON.parse(jsonStr);
        const mems = parsed.memories || [];
        if (mems.length === 0) { r.className = 'err'; r.textContent = '❌ 没有找到 memories 字段，请确认这是从 /export/memories 导出的文件'; p.innerHTML = ''; return; }
        
        pendingJsonData = parsed;
        let html = '<p><b>预览：共 ' + mems.length + ' 条记忆</b></p>';
        const show = mems.slice(0, 10);
        show.forEach(m => { html += '<div class="preview-item">权重 ' + (m.importance || '?') + ' | ' + (m.content || '').substring(0, 80) + '</div>'; });
        if (mems.length > 10) html += '<div class="preview-item" style="color:#999;">...还有 ' + (mems.length - 10) + ' 条</div>';
        html += '<br><button onclick="confirmJsonImport()">确认导入</button>';
        p.innerHTML = html;
        r.textContent = ''; r.className = '';
    } catch(e) { r.className = 'err'; r.textContent = '❌ JSON 格式错误：' + e.message; p.innerHTML = ''; }
}

async function confirmJsonImport() {
    const r = document.getElementById('result');
    if (!pendingJsonData) { r.className = 'err'; r.textContent = '请先预览'; return; }
    
    r.className = 'info'; r.textContent = '导入中...';
    try {
        const resp = await fetch('/import/memories', {
            method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(pendingJsonData)
        });
        const data = await resp.json();
        if (data.error) { r.className = 'err'; r.textContent = '❌ ' + data.error; }
        else { r.className = 'ok'; r.textContent = '✅ 导入完成！新增 ' + data.imported + ' 条，跳过 ' + data.skipped + ' 条（已存在），总计 ' + data.total + ' 条'; }
        document.getElementById('jsonPreview').innerHTML = '';
        pendingJsonData = null;
    } catch(e) { r.className = 'err'; r.textContent = '❌ 请求失败：' + e.message; }
}
</script></body></html>
""")


@app.get("/manage/memories", response_class=HTMLResponse)
async def manage_memories_page():
    """记忆管理页面"""
    if not MEMORY_ENABLED:
        return HTMLResponse("<h3>记忆系统未启用（设置 MEMORY_ENABLED=true 开启）</h3>")
    
    return HTMLResponse("""
<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>管理记忆</title>
<style>
    body { font-family: sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; }
    .toolbar { display: flex; gap: 10px; align-items: center; margin-bottom: 15px; flex-wrap: wrap; }
    input[type="text"] { padding: 8px 12px; font-size: 14px; border: 1px solid #ddd; border-radius: 4px; width: 250px; }
    button { padding: 8px 16px; font-size: 14px; cursor: pointer; border: none; border-radius: 4px; }
    .btn-green { background: #4CAF50; color: white; } .btn-green:hover { background: #45a049; }
    .btn-red { background: #f44336; color: white; } .btn-red:hover { background: #d32f2f; }
    .btn-gray { background: #9e9e9e; color: white; } .btn-gray:hover { background: #757575; }
    table { width: 100%%; border-collapse: collapse; font-size: 14px; }
    th { background: #f5f5f5; padding: 10px 8px; text-align: left; border-bottom: 2px solid #ddd; position: sticky; top: 0; }
    td { padding: 8px; border-bottom: 1px solid #eee; vertical-align: top; }
    tr:hover { background: #fafafa; }
    .content-cell { max-width: 450px; word-break: break-all; }
    .importance-input { width: 45px; padding: 4px; text-align: center; border: 1px solid #ddd; border-radius: 3px; }
    .content-input { width: 100%%; padding: 4px; border: 1px solid #ddd; border-radius: 3px; font-size: 13px; min-height: 40px; resize: vertical; }
    .actions button { padding: 4px 8px; font-size: 12px; margin: 2px; }
    .msg { padding: 10px; margin-bottom: 10px; border-radius: 4px; }
    .ok { background: #e8f5e9; } .err { background: #ffebee; } .info { background: #e3f2fd; }
    .stats { color: #666; font-size: 14px; margin-bottom: 10px; }
    .nav { margin-bottom: 15px; font-size: 14px; color: #666; }
    .nav a { color: #4CAF50; text-decoration: none; }
    .check-col { width: 30px; text-align: center; }
    .id-col { width: 40px; }
    .imp-col { width: 60px; }
    .source-col { width: 90px; font-size: 12px; color: #888; }
    .actions-col { width: 120px; }
</style></head><body>
<h2>🧠 记忆管理</h2>
<div class="nav"><a href="/import/memories">→ 导入新记忆</a> ｜ <a href="/export/memories">→ 导出备份</a></div>

<div class="toolbar">
    <input type="text" id="searchBox" placeholder="搜索记忆..." oninput="filterAndSort()">
    <select id="sortSelect" onchange="filterAndSort()" style="padding:8px 12px;font-size:14px;border:1px solid #ddd;border-radius:4px;">
        <option value="id-desc">ID 从新到旧</option>
        <option value="id-asc">ID 从旧到新</option>
        <option value="imp-desc">权重 从高到低</option>
        <option value="imp-asc">权重 从低到高</option>
    </select>
    <button class="btn-green" onclick="batchSave()">批量保存全部</button>
    <button class="btn-red" onclick="batchDelete()">批量删除选中</button>
    <label style="font-size:13px;color:#666;cursor:pointer;"><input type="checkbox" id="selectAll" onchange="toggleAll()"> 全选</label>
</div>
<div id="msg"></div>
<div class="stats" id="stats"></div>
<div style="overflow-x: auto;">
<table>
    <thead><tr>
        <th class="check-col"><input type="checkbox" id="selectAllHead" onchange="toggleAll()"></th>
        <th class="id-col">ID</th>
        <th>内容</th>
        <th class="imp-col">权重</th>
        <th class="source-col">来源</th>
        <th class="actions-col">操作</th>
    </tr></thead>
    <tbody id="tbody"></tbody>
</table>
</div>

<script>
let allMemories = [];

async function loadMemories() {
    try {
        const resp = await fetch('/api/memories');
        const data = await resp.json();
        allMemories = data.memories || [];
        document.getElementById('stats').textContent = '共 ' + allMemories.length + ' 条记忆';
        filterAndSort();
    } catch(e) { showMsg('err', '加载失败：' + e.message); }
}

function renderTable(mems) {
    const tbody = document.getElementById('tbody');
    tbody.innerHTML = mems.map(m => '<tr data-id="' + m.id + '">' +
        '<td class="check-col"><input type="checkbox" class="mem-check" value="' + m.id + '"></td>' +
        '<td class="id-col">' + m.id + '</td>' +
        '<td class="content-cell"><textarea class="content-input" id="c_' + m.id + '">' + escHtml(m.content) + '</textarea></td>' +
        '<td><input type="number" class="importance-input" id="i_' + m.id + '" value="' + m.importance + '" min="1" max="10"></td>' +
        '<td class="source-col">' + (m.source_session || '-') + '</td>' +
        '<td class="actions"><button class="btn-green" onclick="saveMem(' + m.id + ')">保存</button><button class="btn-red" onclick="delMem(' + m.id + ')">删除</button></td>' +
        '</tr>').join('');
}

function escHtml(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }

function filterAndSort() {
    const q = document.getElementById('searchBox').value.trim().toLowerCase();
    const sort = document.getElementById('sortSelect').value;
    let mems = allMemories;
    if (q) {
        mems = mems.filter(m => m.content.toLowerCase().includes(q));
    }
    mems = [...mems].sort((a, b) => {
        if (sort === 'id-desc') return b.id - a.id;
        if (sort === 'id-asc') return a.id - b.id;
        if (sort === 'imp-desc') return b.importance - a.importance || b.id - a.id;
        if (sort === 'imp-asc') return a.importance - b.importance || a.id - b.id;
        return 0;
    });
    renderTable(mems);
    if (q) document.getElementById('stats').textContent = '搜索到 ' + mems.length + ' / ' + allMemories.length + ' 条';
}

async function saveMem(id) {
    const content = document.getElementById('c_' + id).value;
    const importance = parseInt(document.getElementById('i_' + id).value);
    try {
        const resp = await fetch('/api/memories/' + id, {
            method: 'PUT', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({content, importance})
        });
        const data = await resp.json();
        if (data.error) showMsg('err', '❌ ' + data.error);
        else { showMsg('ok', '✅ 已保存 #' + id); loadMemories(); }
    } catch(e) { showMsg('err', '❌ ' + e.message); }
}

async function delMem(id) {
    if (!confirm('确定删除 #' + id + '？此操作不可撤销。')) return;
    try {
        const resp = await fetch('/api/memories/' + id, { method: 'DELETE' });
        const data = await resp.json();
        if (data.error) showMsg('err', '❌ ' + data.error);
        else { showMsg('ok', '✅ 已删除 #' + id); loadMemories(); }
    } catch(e) { showMsg('err', '❌ ' + e.message); }
}

async function batchSave() {
    const rows = document.querySelectorAll('#tbody tr');
    if (rows.length === 0) { showMsg('err', '没有记忆可保存'); return; }
    const updates = [];
    rows.forEach(row => {
        const id = parseInt(row.dataset.id);
        const cEl = document.getElementById('c_' + id);
        const iEl = document.getElementById('i_' + id);
        if (cEl && iEl) updates.push({id, content: cEl.value, importance: parseInt(iEl.value)});
    });
    if (!confirm('确定保存全部 ' + updates.length + ' 条记忆的修改？')) return;
    try {
        const resp = await fetch('/api/memories/batch-update', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({updates: updates})
        });
        const data = await resp.json();
        if (data.error) showMsg('err', '❌ ' + data.error);
        else { showMsg('ok', '✅ 已保存 ' + data.updated + ' 条'); loadMemories(); }
    } catch(e) { showMsg('err', '❌ ' + e.message); }
}

async function batchDelete() {
    const checked = [...document.querySelectorAll('.mem-check:checked')].map(c => parseInt(c.value));
    if (checked.length === 0) { showMsg('err', '请先勾选要删除的记忆'); return; }
    if (!confirm('确定删除选中的 ' + checked.length + ' 条记忆？此操作不可撤销。')) return;
    try {
        const resp = await fetch('/api/memories/batch-delete', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ids: checked})
        });
        const data = await resp.json();
        if (data.error) showMsg('err', '❌ ' + data.error);
        else { showMsg('ok', '✅ 已删除 ' + data.deleted + ' 条'); loadMemories(); }
    } catch(e) { showMsg('err', '❌ ' + e.message); }
}

function toggleAll() {
    const val = event.target.checked;
    document.querySelectorAll('.mem-check').forEach(c => c.checked = val);
    document.getElementById('selectAll').checked = val;
    document.getElementById('selectAllHead').checked = val;
}

function showMsg(cls, text) {
    const el = document.getElementById('msg');
    el.className = 'msg ' + cls;
    el.textContent = text;
    setTimeout(() => { el.textContent = ''; el.className = ''; }, 4000);
}

loadMemories();
</script></body></html>
""")


# ============================================================
# 管理 API
# ============================================================

@app.get("/api/memories")
async def api_get_memories():
    """获取所有记忆（管理页面用）"""
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    memories = await get_all_memories_detail()
    for m in memories:
        if m.get("created_at"):
            m["created_at"] = str(m["created_at"])
    return {"memories": memories}


@app.put("/api/memories/{memory_id}")
async def api_update_memory(memory_id: int, request: Request):
    """更新单条记忆"""
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    data = await request.json()
    await update_memory(
        memory_id,
        content=data.get("content"),
        importance=data.get("importance"),
    )
    return {"status": "ok", "id": memory_id}


@app.delete("/api/memories/{memory_id}")
async def api_delete_memory(memory_id: int):
    """删除单条记忆"""
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    await delete_memory(memory_id)
    return {"status": "ok", "id": memory_id}


@app.post("/api/memories/batch-update")
async def api_batch_update(request: Request):
    """批量更新记忆"""
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    data = await request.json()
    updates = data.get("updates", [])
    if not updates:
        return {"error": "没有要更新的记忆"}
    for item in updates:
        await update_memory(
            item["id"],
            content=item.get("content"),
            importance=item.get("importance"),
        )
    return {"status": "ok", "updated": len(updates)}


@app.post("/api/memories/batch-delete")
async def api_batch_delete(request: Request):
    """批量删除记忆"""
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    data = await request.json()
    ids = data.get("ids", [])
    if not ids:
        return {"error": "未选择记忆"}
    await delete_memories_batch(ids)
    return {"status": "ok", "deleted": len(ids)}


@app.post("/import/text")
async def import_text_memories(request: Request):
    """从纯文本导入记忆（每行一条），可选自动评分"""
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用（设置 MEMORY_ENABLED=true 开启）"}
    
    try:
        data = await request.json()
        lines = data.get("lines", [])
        skip_scoring = data.get("skip_scoring", False)
        
        if not lines:
            return {"error": "没有找到记忆条目"}
        
        if skip_scoring:
            scored = [{"content": t, "importance": 5} for t in lines]
        else:
            scored = await score_memories(lines)
        
        imported = 0
        skipped = 0
        
        for mem in scored:
            content = mem.get("content", "")
            if not content:
                continue
            
            pool = await get_pool()
            async with pool.acquire() as conn:
                existing = await conn.fetchval(
                    "SELECT COUNT(*) FROM memories WHERE content = $1", content
                )
            
            if existing > 0:
                skipped += 1
                continue
            
            await save_memory(
                content=content,
                importance=mem.get("importance", 5),
                source_session="text-import",
            )
            imported += 1
        
        total = await get_all_memories_count()
        return {
            "status": "done",
            "imported": imported,
            "skipped": skipped,
            "total": total,
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/import/memories")
async def import_memories(request: Request):
    """从 JSON 导入记忆（用于迁移或恢复备份）"""
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用（设置 MEMORY_ENABLED=true 开启）"}
    
    try:
        data = await request.json()
        memories = data.get("memories", [])
        
        if not memories:
            return {"error": "没有找到记忆数据，请确认 JSON 格式正确"}
        
        imported = 0
        skipped = 0
        
        for mem in memories:
            content = mem.get("content", "")
            if not content:
                continue
            
            pool = await get_pool()
            async with pool.acquire() as conn:
                existing = await conn.fetchval(
                    "SELECT COUNT(*) FROM memories WHERE content = $1", content
                )
            
            if existing > 0:
                skipped += 1
                continue
            
            await save_memory(
                content=content,
                importance=mem.get("importance", 5),
                source_session=mem.get("source_session", "json-import"),
            )
            imported += 1
        
        total = await get_all_memories_count()
        return {
            "status": "done",
            "imported": imported,
            "skipped": skipped,
            "total": total,
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================

if __name__ == "__main__":
    import uvicorn
    print(f"🚀 AI Memory Gateway 启动中... 端口 {PORT}")
    print(f"📝 人设长度：{len(SYSTEM_PROMPT)} 字符")
    print(f"🤖 默认模型：{DEFAULT_MODEL}")
    print(f"🔗 API 地址：{API_BASE_URL}")
    print(f"🧠 记忆系统：{'开启' if MEMORY_ENABLED else '关闭'}")
    print(f"🔄 记忆提取间隔：{'禁用' if MEMORY_EXTRACT_INTERVAL == 0 else '每轮提取' if MEMORY_EXTRACT_INTERVAL == 1 else f'每 {MEMORY_EXTRACT_INTERVAL} 轮提取一次'}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
