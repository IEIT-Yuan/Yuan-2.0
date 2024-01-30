"""
This script implements an API for the Yuan2 model,
formatted similarly to OpenAI's API (https://platform.openai.com/docs/api-reference/chat).
It's designed to be run as a web server using FastAPI and uvicorn,
making the Yuan2 model accessible through OpenAI Client.

Key Components and Features:
- Model and Tokenizer Setup: Configures the model and tokenizer paths and loads them.
- FastAPI Configuration: Sets up a FastAPI application with CORS middleware for handling cross-origin requests.
- API Endpoints:
  - "/v1/models": Lists the available models, specifically Yuan2.
  - "/v1/chat/completions": Processes chat completion requests with options for streaming and regular responses.
  - "/v1/embeddings": Processes Embedding request of a list of text inputs.
- Token Limit Caution: In the Yuan2.0 API, 'max_tokens' is equivalent to HuggingFace's 'max_new_tokens', not 'max_length'.
For instance, setting 'max_tokens' to 8192 for a 2B model would result in an error due to the model's inability to output
that many tokens after accounting for the history and prompt tokens.
- Stream Handling and Custom Functions: Manages streaming responses and custom function calls within chat responses.
- Pydantic Models: Defines structured models for requests and responses, enhancing API documentation and type safety.
- Main Execution: Initializes the model and tokenizer, and starts the FastAPI app on the designated host and port.

Note:
    This script doesn't include the setup for special tokens or multi-GPU support by default.
    Users need to configure their special tokens and can enable multi-GPU support as per the provided instructions.
    Embedding Models only support in One GPU.

"""

import os
import time
import tiktoken
import torch
import uvicorn

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field
from transformers import LlamaTokenizer, AutoModelForCausalLM
from utils import process_response, generate_chatglm3, generate_stream_chatglm3
from sentence_transformers import SentenceTransformer

from sse_starlette.sse import EventSourceResponse
from log import get_logger

# Set up limit request time
EventSourceResponse.DEFAULT_PING_INTERVAL = 1000

# set LLM path
MODEL_NAME = os.environ.get('MODEL_NAME', "yuan2")
MODEL_PATH = os.environ.get('MODEL_PATH', 'IEITYuan/Yuan2-2B-Janus-hf')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)

# set Embedding Model path
EMBEDDING_PATH = os.environ.get('EMBEDDING_PATH', 'BAAI/bge-large-zh-v1.5')


def _gc(forced: bool = False):
    # global args
    # if args.disable_gc and not forced:
    #     return

    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


@asynccontextmanager
async def lifespan(app: FastAPI):  # collects GPU memory
    yield
    _gc(forced=True)


app = FastAPI(lifespan=lifespan)


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "IEITYuan"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class FunctionCallResponse(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "function"]
    content: str = None
    name: Optional[str] = None
    function_call: Optional[FunctionCallResponse] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None
    function_call: Optional[FunctionCallResponse] = None


## for Embedding
class EmbeddingRequest(BaseModel):
    input: List[str]
    model: str


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    data: list
    model: str
    object: str
    usage: CompletionUsage


# for ChatCompletionRequest
class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = None
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    stop: Union[Optional[str], List[str], None] = "<eod>"
    echo: Optional[bool] = False
    n: Optional[int] = 1
    seed: Optional[int]
    tools: Optional[Union[dict, List[dict]]] = None
    repetition_penalty: Optional[float] = 1.2
    frequency_penalty: Optional[float]
    presence_penalty: Optional[float]


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "function_call"]


class ChatCompletionResponseStreamChoice(BaseModel):
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length", "function_call"]]
    index: int


class ChatCompletionResponse(BaseModel):
    model: str
    id: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: Optional[UsageInfo] = None


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    embeddings = [embedding_model.encode(text) for text in request.input]
    embeddings = [embedding.tolist() for embedding in embeddings]

    def num_tokens_from_string(string: str) -> int:
        """
        Returns the number of tokens in a text string.
        use cl100k_base tokenizer
        """
        encoding = tiktoken.get_encoding('cl100k_base')
        num_tokens = len(encoding.encode(string))
        return num_tokens

    response = {
        "data": [
            {
                "object": "embedding",
                "embedding": embedding,
                "index": index
            }
            for index, embedding in enumerate(embeddings)
        ],
        "model": request.model,
        "object": "list",
        "usage": CompletionUsage(
            prompt_tokens=sum(len(text.split()) for text in request.input),
            completion_tokens=0,
            total_tokens=sum(num_tokens_from_string(text) for text in request.input),
        )
    }
    return response


# A global registry for all conversation templates
model_list: Dict[str, ModelCard] = {}


def register_model_list(model_card: ModelCard):
    """Register a new model card in model list, e.g.:
    register_model_list(
        ModelCard(
            id="yuan2-2b-janus",
        )
    )
    """
    id = model_card.id.lower()
    assert (
            id in model_list
    ), f"{id} has been registered, register info: {model_list[id]}."

    model_list[id] = model_card


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    return ModelList(
        data=model_list
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer

    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid request")

    exclude = {
        "n",
        "best_of",
        "user",
        "messages",
    }
    request.model_dump(exclude=exclude)

    gen_params = dict(
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        max_tokens=request.max_tokens or 1024,
        echo=False,
        stream=request.stream,
        stop=request.stop,
        repetition_penalty=request.repetition_penalty,
        functions=request.functions,
        tools=request.tools,
    )
    logger.debug(f"==== request ====\n{gen_params}")

    if request.stream:
        if request.functions:
            raise HTTPException(
                status_code=400,
                detail=
                'Invalid request: Function calling is not yet implemented for stream mode.',
            )

        # Use the stream mode to read the first few characters, if it is not a function call, direct stram output
        predict_stream_generator = predict_stream(request.model, gen_params)
        output = next(predict_stream_generator)
        if not contains_custom_function(output):
            return EventSourceResponse(predict_stream_generator, media_type="text/event-stream")

        # Obtain the result directly at one time and determine whether tools needs to be called.
        logger.debug(f"First result output：\n{output}")

        function_call = None
        if output and request.tools:
            try:
                function_call = process_response(output, use_tool=True)
            except:
                logger.warning("Failed to parse tool call")

        # CallFunction
        if isinstance(function_call, dict):
            function_call = FunctionCallResponse(**function_call)

            """
            In this demo, we did not register any tools.
            You can use the tools that have been implemented in our `tools_using_demo` and implement your own streaming tool implementation here.
            Similar to the following method:
                function_args = json.loads(function_call.arguments)
                tool_response = dispatch_tool(tool_name: str, tool_params: dict)
            """
            tool_response = ""

            if not gen_params.get("messages"):
                gen_params["messages"] = []

            gen_params["messages"].append(ChatMessage(
                role="assistant",
                content=output,
            ))
            gen_params["messages"].append(ChatMessage(
                role="function",
                name=function_call.name,
                content=tool_response,
            ))

            # Streaming output of results after function calls
            generate = predict(request.model, gen_params)
            return EventSourceResponse(generate, media_type="text/event-stream")

        else:
            # Handled to avoid exceptions in the above parsing function process.
            generate = parse_output_text(request.model, output)
            return EventSourceResponse(generate, media_type="text/event-stream")

    # Here is the handling of stream = False
    response = generate_chatglm3(model, tokenizer, gen_params)

    # Remove the first newline character
    if response["text"].startswith("\n"):
        response["text"] = response["text"][1:]
    response["text"] = response["text"].strip()

    usage = UsageInfo()
    function_call, finish_reason = None, "stop"
    if request.tools:
        try:
            function_call = process_response(response["text"], use_tool=True)
        except:
            logger.warning("Failed to parse tool call, maybe the response is not a tool call or have been answered.")

    if isinstance(function_call, dict):
        finish_reason = "function_call"
        function_call = FunctionCallResponse(**function_call)

    message = ChatMessage(
        role="assistant",
        content=response["text"],
        function_call=function_call if isinstance(function_call, FunctionCallResponse) else None,
    )

    logger.debug(f"==== message ====\n{message}")

    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=message,
        finish_reason=finish_reason,
    )
    task_usage = UsageInfo.model_validate(response["usage"])
    for usage_key, usage_value in task_usage.model_dump().items():
        setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

    return ChatCompletionResponse(
        model=request.model,
        id="",  # for open_source model, id is empty
        choices=[choice_data],
        object="chat.completion",
        usage=usage
    )



async def chat_completion_stream_generator(
    model_name: str, gen_params: Dict[str, Any], n: int, worker_addr: str
) -> Generator[str, Any, None]:
    """
    Event stream format:
    https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format
    """
    id = f"chatcmpl-{shortuuid.random()}"
    finish_stream_events = []
    for i in range(n):
        # First chunk with role
        choice_data = ChatCompletionResponseStreamChoice(
            index=i,
            delta=DeltaMessage(role="assistant"),
            finish_reason=None,
        )
        chunk = ChatCompletionStreamResponse(
            id=id, choices=[choice_data], model=model_name
        )
        yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"

        previous_text = ""
        async for content in generate_completion_stream(gen_params, worker_addr):
            if content["error_code"] != 0:
                yield f"data: {json.dumps(content, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            decoded_unicode = content["text"].replace("\ufffd", "")
            delta_text = decoded_unicode[len(previous_text) :]
            previous_text = (
                decoded_unicode
                if len(decoded_unicode) > len(previous_text)
                else previous_text
            )

            if len(delta_text) == 0:
                delta_text = None
            choice_data = ChatCompletionResponseStreamChoice(
                index=i,
                delta=DeltaMessage(content=delta_text),
                finish_reason=content.get("finish_reason", None),
            )
            chunk = ChatCompletionStreamResponse(
                id=id, choices=[choice_data], model=model_name
            )
            if delta_text is None:
                if content.get("finish_reason", None) is not None:
                    finish_stream_events.append(chunk)
                continue
            yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"
    # There is not "content" field in the last delta message, so exclude_none to exclude field "content".
    for finish_chunk in finish_stream_events:
        yield f"data: {finish_chunk.json(exclude_none=True, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"



async def predict(model_id: str, params: dict):
    global model, tokenizer

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, id="", choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    previous_text = ""
    for new_response in generate_stream_chatglm3(model, tokenizer, params):
        decoded_unicode = new_response["text"]
        delta_text = decoded_unicode[len(previous_text):]
        previous_text = decoded_unicode

        finish_reason = new_response["finish_reason"]
        if len(delta_text) == 0 and finish_reason != "function_call":
            continue

        function_call = None
        if finish_reason == "function_call":
            try:
                function_call = process_response(decoded_unicode, use_tool=True)
            except:
                logger.warning(
                    "Failed to parse tool call, maybe the response is not a tool call or have been answered.")

        if isinstance(function_call, dict):
            function_call = FunctionCallResponse(**function_call)

        delta = DeltaMessage(
            content=delta_text,
            role="assistant",
            function_call=function_call if isinstance(function_call, FunctionCallResponse) else None,
        )

        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=delta,
            finish_reason=finish_reason
        )
        chunk = ChatCompletionResponse(
            model=model_id,
            id="",
            choices=[choice_data],
            object="chat.completion.chunk"
        )
        yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(
        model=model_id,
        id="",
        choices=[choice_data],
        object="chat.completion.chunk"
    )
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    yield '[DONE]'


def predict_stream(model_id, gen_params):
    """
    The function call is compatible with stream mode output.

    The first seven characters are determined.
    If not a function call, the stream output is directly generated.
    Otherwise, the complete character content of the function call is returned.

    :param model_id:
    :param gen_params:
    :return:
    """
    output = ""
    is_function_call = False
    has_send_first_chunk = False
    for new_response in generate_stream_chatglm3(model, tokenizer, gen_params):
        decoded_unicode = new_response["text"]
        delta_text = decoded_unicode[len(output):]
        output = decoded_unicode

        # When it is not a function call and the character length is> 7,
        # try to judge whether it is a function call according to the special function prefix
        if not is_function_call and len(output) > 7:

            # Determine whether a function is called
            is_function_call = contains_custom_function(output)
            if is_function_call:
                continue

            # Non-function call, direct stream output
            finish_reason = new_response["finish_reason"]

            # Send an empty string first to avoid truncation by subsequent next() operations.
            if not has_send_first_chunk:
                message = DeltaMessage(
                    content="",
                    role="assistant",
                    function_call=None,
                )
                choice_data = ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=message,
                    finish_reason=finish_reason
                )
                chunk = ChatCompletionResponse(
                    model=model_id,
                    id="",
                    choices=[choice_data],
                    created=int(time.time()),
                    object="chat.completion.chunk"
                )
                yield "{}".format(chunk.model_dump_json(exclude_unset=True))

            send_msg = delta_text if has_send_first_chunk else output
            has_send_first_chunk = True
            message = DeltaMessage(
                content=send_msg,
                role="assistant",
                function_call=None,
            )
            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=message,
                finish_reason=finish_reason
            )
            chunk = ChatCompletionResponse(
                model=model_id,
                id="",
                choices=[choice_data],
                created=int(time.time()),
                object="chat.completion.chunk"
            )
            yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    if is_function_call:
        yield output
    else:
        yield '[DONE]'


async def parse_output_text(model_id: str, value: str):
    """
    Directly output the text content of value

    :param model_id:
    :param value:
    :return:
    """
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant", content=value),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, id="", choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(model=model_id, id="", choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    yield '[DONE]'


def contains_custom_function(value: str) -> bool:
    """
    Determine whether 'function_call' according to a special function prefix.

    For example, the functions defined in "tools_using_demo/tool_register.py" are all "get_xxx" and start with "get_"

    [Note] This is not a rigorous judgment method, only for reference.

    :param value:
    :return:
    """
    return value and 'get_' in value


def load_chat_template(tokenizer):
    return tokenizer.chat_template


def load_model_and_tokenizer(model_path: str, device_map):
    tokenizer = LlamaTokenizer.from_pretrained(
        model_path,
        add_eos_token=False,
        add_bos_token=False,
        eos_token='<eod>',
    )
    tokenizer.add_tokens(
        [
            '<sep>',
            '<pad>',
            '<mask>',
            '<predict>',
            '<FIM_SUFFIX>',
            '<FIM_PREFIX>',
            '<FIM_MIDDLE>',
            '<commit_before>',
            '<commit_msg>',
            '<commit_after>',
            '<jupyter_start>',
            '<jupyter_text>',
            '<jupyter_code>',
            '<jupyter_output>',
            '<empty_output>'
        ],
        special_tokens=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    return model, tokenizer


def generate_prompts(tokenizer, messages):
    # Call the function and get the result
    result = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=False,
    )

    return result


def _validate_prompt_and_tokenize(
        request: Union[ChatCompletionRequest, CompletionRequest],
        prompt: Optional[str] = None,
        prompt_ids: Optional[List[int]] = None) -> List[int]:
    if not (prompt or prompt_ids):
        raise ValueError("Either prompt or prompt_ids should be provided.")
    if (prompt and prompt_ids):
        raise ValueError(
            "Only one of prompt or prompt_ids should be provided.")

    input_ids = prompt_ids if prompt_ids is not None else self.tokenizer(
        prompt).input_ids
    token_num = len(input_ids)

    if request.max_tokens is None:
        request.max_tokens = self.max_model_len - token_num

    if token_num + request.max_tokens > self.max_model_len:
        raise ValueError(
            f"This model's maximum context length is {self.max_model_len} tokens. "
            f"However, you requested {request.max_tokens + token_num} tokens "
            f"({token_num} in the messages, "
            f"{request.max_tokens} in the completion). "
            f"Please reduce the length of the messages or completion.", )
    else:
        return input_ids


def _get_args():
    parser = argparse.ArgumentParser(
        description="OpenAI-Compatible RESTful API server for Yuan2.0."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="yuan2-2b-janus-hf",
        help="a short name for model info."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="IEITYuan/Yuan2-2B-Janus-hf",
        help="full model name or local path."
    )
    parser.add_argument(
        "--cpu-only",
        type=bool,
        default=False,
        help='Run demo with CPU only'
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="host name"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="port number"
    )
    parser.add_argument(
        "--allow-credentials",
        action="store_true",
        help="allow credentials"
    )
    parser.add_argument(
        "--allowed-origins",
        type=json.loads,
        default=["*"],
        help="allowed origins"
    )
    parser.add_argument(
        "--allowed-methods",
        type=json.loads,
        default=["*"],
        help="allowed methods"
    )
    parser.add_argument(
        "--allowed-headers",
        type=json.loads,
        default=["*"],
        help="allowed headers"
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=1024,
        help='Sets the default maximum token length for the prompt + response (defaults to 1024)'
    )
    parser.add_argument(
        '--tensor-parallel',
        type=int,
        default=1,
        help='Number of GPUs to split the model across (defaults to 1)'
    )
    parser.add_argument(
        '--replica-num',
        type=int,
        default=1,
        help='The number of model replicas to stand up (defaults to 1)'
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default=None,
        help="Path to chat template file,, or chat template string (defaults to None)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        help="log level"
    )

    args = parser.parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    logger.info(f"args: {args}")
    return args


if __name__ == "__main__":

    args = _get_args()

    # 注册model_list
    register_model_list(ModelCard(
        id=args.model_name,
    ))

    if args.cpu_only:
        device_map = 'cpu'
    else:
        device_map = 'auto'

    # Load LLM
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path, device_map)


    # load Embedding
    embedding_model = SentenceTransformer(EMBEDDING_PATH, device="cuda")
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
