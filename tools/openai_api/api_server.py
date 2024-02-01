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

"""
import argparse
import asyncio
import codecs
import os
import json
import random

import numpy as np
import shortuuid as shortuuid
import tiktoken
import time
import torch
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sse_starlette.sse import EventSourceResponse
from transformers import LlamaTokenizer, AutoModelForCausalLM
from typing import List, Literal, Optional, Union, Dict, Any, Generator, Iterator, Iterable

from constants import ErrorCode
from log import get_logger

logger = get_logger("debug")

# Set up limit request time
EventSourceResponse.DEFAULT_PING_INTERVAL = 1000

# set LLM path
MODEL_NAME = os.environ.get('MODEL_NAME', "yuan2")
MODEL_PATH = os.environ.get('MODEL_PATH', 'IEITYuan/Yuan2-2B-Janus-hf')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)

# set Embedding Model path
EMBEDDING_PATH = os.environ.get('EMBEDDING_PATH', 'BAAI/bge-large-zh-v1.5')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@asynccontextmanager
async def lifespan(app: FastAPI):  # collects GPU memory
    yield
    _gc()


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
    stop: Union[Optional[str], List[str], None] = ["<eod>"]
    echo: Optional[bool] = False
    n: Optional[int] = 1
    seed: Optional[int]
    repetition_penalty: Optional[float] = 1.2

    # Not support yet.
    tools: Optional[Union[dict, List[dict]]] = None
    tool_choice: Optional[Union[str, dict]] = None
    logit_bias: Optional[Dict[str, int]] = None
    logprobs: Optional[bool] = False
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    user: Optional[str] = None


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


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    code: int


def create_error_response(code: int, message: str) -> JSONResponse:
    return JSONResponse(
        ErrorResponse(message=message, code=code).model_dump(), status_code=400
    )


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
model_map: Dict[str, ModelCard] = {}


def register_model_info(model_card: ModelCard):
    """Register a new model card in model list, e.g.:
    register_model_info(
        ModelCard(
            id="yuan2-2b-janus",
        )
    )
    """
    id = model_card.id.lower()
    assert (
            id not in model_map.keys()
    ), f"{id} has been registered, register info: {model_map[id]}."

    model_map[id] = model_card


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    return ModelList(
        data=model_map.values()
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer
    print(request)

    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid request")

    # request = await handle_request(request)

    # if request.seed is not None:
    #     set_seed(request.seed)

    exclude = {
        "model",
        "messages",
        "seed",
        "stream",

        # todo: not support yet.
        "tools",
        "tool_choice",
        "logit_bias",
        "logprobs",
        "frequency_penalty",
        "presence_penalty",
        "user",
    }

    kwargs = request.model_dump(exclude=exclude)
    prompts = tokenizer.apply_chat_template(request.messages, tokenize=False)

    kwargs["prompt"] = prompts
    kwargs["max_new_tokens"] = request.max_tokens
    kwargs["do_sample"] = True

    logger.debug(f"==== request params ====\n{kwargs}")

    # if request.stream:
    #     generator = chat_completion_stream_generator(request.model, kwargs, request.n)
    #     return StreamingResponse(generator, media_type="text/event-stream")

    choices = []
    chat_completions = []
    for _ in range(request.n):
        content = asyncio.create_task(generate_completion(kwargs))
        chat_completions.append(content)
    try:
        all_tasks = await asyncio.gather(*chat_completions)
    except Exception as e:
        return create_error_response(ErrorCode.INTERNAL_ERROR, str(e))

    usage = UsageInfo()
    for i, content in enumerate(all_tasks):
        choices.append(
            ChatCompletionResponseChoice(
                index=i,
                message=ChatMessage(role="assistant", content=content["text"]),
                finish_reason=content.get("finish_reason", "stop"),
            )
        )
        if "usage" in content:
            task_usage = UsageInfo.parse_obj(content["usage"])
            for usage_key, usage_value in task_usage.dict().items():
                setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

    return ChatCompletionResponse(model=request.model, choices=choices, usage=usage)

#
# def _create_chat_completion_generator(**kwargs) -> Union[Iterator[ChatCompletionResponse], ChatCompletionResponse]:
#     return (
#         _create_chat_completion_stream(kwargs)
#         if kwargs.get("stream", False)
#         else _create_chat_completion(kwargs)
#     )

#
# def _create_chat_completion_stream(params: Dict[str, Any]) -> Iterator[ChatCompletionResponse]:
#     """
#     Creates a chat completion stream.
#
#     Args:
#         params (Dict[str, Any]): The parameters for generating the chat completion.
#
#     Yields:
#         Dict[str, Any]: The output of the chat completion stream.
#     """
#     _id, _created, _model = None, None, None
#     has_function_call = False
#     for i, output in enumerate(self._generate(params)):
#         if output["error_code"] != 0:
#             yield output
#             return
#
#         _id, _created, _model = output["id"], output["created"], output["model"]
#         if i == 0:
#             choice = ChunkChoice(
#                 index=0,
#                 delta=ChoiceDelta(role="assistant", content=""),
#                 finish_reason=None,
#                 logprobs=None,
#             )
#             yield ChatCompletionChunk(
#                 id=f"chat{_id}",
#                 choices=[choice],
#                 created=_created,
#                 model=_model,
#                 object="chat.completion.chunk",
#             )
#
#         finish_reason = output["finish_reason"]
#         if len(output["delta"]) == 0 and finish_reason != "function_call":
#             continue
#
#         function_call = None
#         if finish_reason == "function_call":
#             try:
#                 _, function_call = self.prompt_adapter.parse_assistant_response(
#                     output["text"], params.get("functions"), params.get("tools"),
#                 )
#             except Exception as e:
#                 traceback.print_exc()
#                 logger.warning("Failed to parse tool call")
#
#         if isinstance(function_call, dict) and "arguments" in function_call:
#             has_function_call = True
#             function_call = ChoiceDeltaFunctionCall(**function_call)
#             delta = ChoiceDelta(
#                 content=output["delta"],
#                 function_call=function_call
#             )
#         elif isinstance(function_call, dict) and "function" in function_call:
#             has_function_call = True
#             finish_reason = "tool_calls"
#             function_call["index"] = 0
#             tool_calls = [model_parse(ChoiceDeltaToolCall, function_call)]
#             delta = ChoiceDelta(
#                 content=output["delta"],
#                 tool_calls=tool_calls,
#             )
#         else:
#             delta = ChoiceDelta(content=output["delta"])
#
#         choice = ChunkChoice(
#             index=0,
#             delta=delta,
#             finish_reason=finish_reason,
#             logprobs=None,
#         )
#         yield ChatCompletionChunk(
#             id=f"chat{_id}",
#             choices=[choice],
#             created=_created,
#             model=_model,
#             object="chat.completion.chunk",
#         )
#
#     if not has_function_call:
#         choice = ChunkChoice(
#             index=0,
#             delta=ChoiceDelta(),
#             finish_reason="stop",
#             logprobs=None,
#         )
#         yield ChatCompletionChunk(
#             id=f"chat{_id}",
#             choices=[choice],
#             created=_created,
#             model=_model,
#             object="chat.completion.chunk",
#         )

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if torch.is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


async def handle_request(request: ChatCompletionRequest) -> Union[ChatCompletionRequest, JSONResponse]:
    error = check_requests(request)
    if error is not None:
        return error

    request.stop = request.stop or []
    if isinstance(request.stop, str):
        request.stop = [request.stop]

    request.top_p = max(request.top_p, 1e-5)
    if request.temperature <= 1e-5:
        request.top_p = 1.0

    return request


def check_requests(request: ChatCompletionRequest) -> Optional[JSONResponse]:
    # Check all params
    if request.model not in model_map.keys():
        return create_error_response(
            ErrorCode.INVALID_MODEL,
            f"Model '{request.model}' not found. Please use one of '{model_map.keys()}'.",
        )
    if request.max_tokens is not None and request.max_tokens <= 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.max_tokens} is less than the minimum of 1 - 'max_tokens'",
        )
    if request.n is not None and request.n <= 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.n} is less than the minimum of 1 - 'n'",
        )
    if request.temperature is not None and request.temperature < 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is less than the minimum of 0 - 'temperature'",
        )
    if request.temperature is not None and request.temperature > 2:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is greater than the maximum of 2 - 'temperature'",
        )
    if request.top_p is not None and request.top_p < 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is less than the minimum of 0 - 'top_p'",
        )
    if request.top_p is not None and request.top_p > 1:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is greater than the maximum of 1 - 'temperature'",
        )
    if request.repetition_penalty is not None and request.repetition_penalty < 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is less than the minimum of 0 - 'repetition_penalty'",
        )
    if request.stop is None or isinstance(request.stop, (str, list)):
        return None
    else:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.stop} is not valid under any of the given schemas - 'stop'",
        )


# async def chat_completion_stream_generator(
#         model_name: str, gen_params: Dict[str, Any], n: int
# ) -> Generator[str, Any, None]:
#     """
#     Event stream format:
#     https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format
#     """
#     id = f"chatcmpl-{shortuuid.random()}"
#     finish_stream_events = []
#     for i in range(n):
#         # First chunk with role
#         choice_data = ChatCompletionResponseStreamChoice(
#             index=i,
#             delta=DeltaMessage(role="assistant"),
#             finish_reason=None,
#         )
#         chunk = ChatCompletionResponse(
#             id=id, choices=[choice_data], model=model_name
#         )
#         yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"
#
#         previous_text = ""
#         async for content in generate(model, tokenizer, gen_params):
#             if content["error_code"] != 0:
#                 yield f"data: {json.dumps(content, ensure_ascii=False)}\n\n"
#                 yield "data: [DONE]\n\n"
#                 return
#             decoded_unicode = content["text"].replace("\ufffd", "")
#             delta_text = decoded_unicode[len(previous_text):]
#             previous_text = (
#                 decoded_unicode
#                 if len(decoded_unicode) > len(previous_text)
#                 else previous_text
#             )
#
#             if len(delta_text) == 0:
#                 delta_text = None
#             choice_data = ChatCompletionResponseStreamChoice(
#                 index=i,
#                 delta=DeltaMessage(content=delta_text),
#                 finish_reason=content.get("finish_reason", None),
#             )
#             chunk = ChatCompletionResponse(
#                 id=id, choices=[choice_data], model=model_name, object="chat.completion.chunk",
#             )
#             if delta_text is None:
#                 if content.get("finish_reason", None) is not None:
#                     finish_stream_events.append(chunk)
#                 continue
#             yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"
#     # There is not "content" field in the last delta message, so exclude_none to exclude field "content".
#     for finish_chunk in finish_stream_events:
#         yield f"data: {finish_chunk.json(exclude_none=True, ensure_ascii=False)}\n\n"
#     yield "data: [DONE]\n\n"

#
# async def api_generate_stream(model, tokenizer, kwargs):
#     generator = generate_stream(model, tokenizer, kwargs)
#     return StreamingResponse(generator)
#
#
# async def api_generate(model, tokenizer, kwargs):
#     output = await asyncio.to_thread(generate, (model, tokenizer, kwargs))
#     return JSONResponse(output)

#
# def generate(model, tokenizer, kwargs):
#     for x in generate_stream_gate(model, tokenizer, kwargs):
#         pass
#     return json.loads(x[:-1].decode())
#
#
# def generate_stream(model, tokenizer, kwargs):
#     global device
#     try:
#         for output in _generate_stream(
#             model,
#             tokenizer,
#             kwargs,
#             device,
#             context_len,
#             stream_interval,
#         ):
#             ret = {
#                 "text": output["text"],
#                 "error_code": 0,
#             }
#             if "usage" in output:
#                 ret["usage"] = output["usage"]
#             if "finish_reason" in output:
#                 ret["finish_reason"] = output["finish_reason"]
#             if "logprobs" in output:
#                 ret["logprobs"] = output["logprobs"]
#             yield json.dumps(ret).encode() + b"\0"
#     except torch.cuda.OutOfMemoryError as e:
#         ret = {
#             "text": f"cuda Out of memory error: \n\n({e})",
#             "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
#         }
#         yield json.dumps(ret).encode() + b"\0"
#     except (ValueError, RuntimeError) as e:
#         ret = {
#             "text": f"internal server error: \n\n({e})",
#             "error_code": ErrorCode.INTERNAL_ERROR,
#         }
#         yield json.dumps(ret).encode() + b"\0"
#

@torch.inference_mode()
def _generate_stream(
    model,
    tokenizer,
    params: Dict,
    device: str,
    context_len: int,
    stream_interval: int = 2,
    judge_sent_end: bool = False,
):
    if hasattr(model, "device"):
        device = model.device

    # Read parameters
    prompt = params["prompt"]
    len_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", -1))  # -1 means disable
    max_new_tokens = int(params.get("max_new_tokens", 1024))
    logprobs = params.get("logprobs", None)
    echo = bool(params.get("echo", True))
    stop_str = params.get("stop", None) or []
    stop_token_ids = params.get("stop_token_ids", None) or []
    stop_token_ids.update([tokenizer(s) for s in stop_str])

    if tokenizer.eos_token_id not in stop_token_ids:
        stop_token_ids.append(tokenizer.eos_token_id)

    input_ids = tokenizer(prompt).input_ids
    input_ids = input_ids.to(model.device)

    max_src_len = context_len - max_new_tokens - 1

    input_ids = input_ids[-max_src_len:]
    output_ids = list(input_ids)
    input_echo_len = len(input_ids)

    start_ids = torch.as_tensor([input_ids], device=device)

    past_key_values = out = None
    token_logprobs = [None]  # The first token has no logprobs.
    sent_interrupt = False
    finish_reason = None
    stopped = False
    for i in range(max_new_tokens):
        if i == 0:  # prefill
            out = model(input_ids=start_ids, use_cache=True)
            logits = out.logits
            past_key_values = out.past_key_values

            if logprobs is not None:
                # Prefull logprobs for the prompt.
                shift_input_ids = start_ids[..., 1:].contiguous()
                shift_logits = logits[..., :-1, :].contiguous()
                shift_logits = torch.log_softmax(shift_logits, dim=-1).tolist()
                for label_id, logit in zip(
                    shift_input_ids[0].tolist(), shift_logits[0]
                ):
                    token_logprobs.append(logit[label_id])
        else:  # decoding
            out = model(
                input_ids=torch.as_tensor(
                    [[token] if not sent_interrupt else output_ids],
                    device=device,
                ),
                use_cache=True,
                past_key_values=past_key_values if not sent_interrupt else None,
            )
            sent_interrupt = False
            logits = out.logits
            past_key_values = out.past_key_values

        last_token_logits = logits[0, -1, :]

        if temperature < 1e-5 or top_p < 1e-8:  # greedy
            _, indices = torch.topk(last_token_logits, 2)
            tokens = [int(index) for index in indices.tolist()]
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            indices = torch.multinomial(probs, num_samples=2)
            tokens = [int(token) for token in indices.tolist()]
        token = tokens[0]
        output_ids.append(token)
        if logprobs is not None:
            # Cannot use last_token_logits because logprobs is based on raw logits.
            token_logprobs.append(
                torch.log_softmax(logits[0, -1, :], dim=-1)[token].tolist()
            )

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        # Yield the output tokens
        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            if echo:
                tmp_output_ids = output_ids
                rfind_start = len_prompt
            else:
                tmp_output_ids = output_ids[input_echo_len:]
                rfind_start = 0

            output = tokenizer.decode(
                tmp_output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
            ret_logprobs = None
            if logprobs is not None:
                ret_logprobs = {
                    "text_offset": [],
                    "tokens": [
                        tokenizer.decode(token)
                        for token in (
                            output_ids if echo else output_ids[input_echo_len:]
                        )
                    ],
                    "token_logprobs": token_logprobs
                    if echo
                    else token_logprobs[input_echo_len:],
                    "top_logprobs": [{}]
                    * len(token_logprobs if echo else token_logprobs[input_echo_len:]),
                }
                # Compute text_offset
                curr_pos = 0
                for text in ret_logprobs["tokens"]:
                    ret_logprobs["text_offset"].append(curr_pos)
                    curr_pos += len(text)

            if judge_sent_end and stopped and not is_sentence_complete(output):
                if len(tokens) > 1:
                    token = tokens[1]
                    output_ids[-1] = token
                else:
                    output_ids.pop()
                stopped = False
                sent_interrupt = True

            partially_stopped = False
            if stop_str:
                if isinstance(stop_str, str):
                    pos = output.rfind(stop_str, rfind_start)
                    if pos != -1:
                        output = output[:pos]
                        stopped = True
                    else:
                        partially_stopped = is_partial_stop(output, stop_str)
                elif isinstance(stop_str, Iterable):
                    for each_stop in stop_str:
                        pos = output.rfind(each_stop, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
                            break
                        else:
                            partially_stopped = is_partial_stop(output, each_stop)
                            if partially_stopped:
                                break
                else:
                    raise ValueError("Invalid stop field type.")

            # Prevent yielding partial stop sequence
            if not partially_stopped:
                yield {
                    "text": output,
                    "logprobs": ret_logprobs,
                    "usage": {
                        "prompt_tokens": input_echo_len,
                        "completion_tokens": i,
                        "total_tokens": input_echo_len + i,
                    },
                    "finish_reason": None,
                }

        if stopped:
            break

    # Finish stream event, which contains finish reason
    else:
        finish_reason = "length"

    if stopped:
        finish_reason = "stop"

    yield {
        "text": output,
        "logprobs": ret_logprobs,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
        "finish_reason": finish_reason,
    }

    # Clean
    del past_key_values, out
    _gc()


def _gc():
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def is_partial_stop(output: str, stop_str: str):
    """Check whether the output contains a partial stop str."""
    for i in range(0, min(len(output), len(stop_str))):
        if stop_str.startswith(output[-i:]):
            return True
    return False


def is_sentence_complete(output: str):
    """Check whether the output is a complete sentence."""
    end_symbols = (".", "?", "!", "...", "。", "？", "！", "…", '"', "'", "”")
    return output.endswith(end_symbols)



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


def postprocess(output_text, stop):
    # 后处理输出
    # 这里可以实现一些后处理的逻辑，比如去除重复，过滤敏感词，添加标点等
    # 这里只是一个简单的示例，你可以根据你的需要修改它
    output_text = output_text.replace('<unk>', '').replace('▃', '\n').replace('<n>', '\n').replace('▂', ' ')
    for s in stop:
        output_text = output_text.lstrip("<sep>").rstrip(s)
    return output_text


async def generate_completion(kwargs):
    global device
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer(kwargs["prompt"], return_tensors="pt").input_ids.to(device)

        # 生成输出
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=kwargs["max_tokens"],
            temperature=kwargs["temperature"],
            top_k=kwargs["top_k"],
            top_p=kwargs["top_p"],
            repetition_penalty=kwargs["repetition_penalty"],
            do_sample=kwargs["do_sample"],
        )
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)  # 对输出进行解码
        output_text = postprocess(output_text, kwargs["stop"])  # 后处理输出
    return output_text  # 返回输出


def stream_inference(model, tokenizer, kwargs):
    input_text = kwargs["messages"]
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

    # 初始化生成的文本
    generated_text = ""

    # 开始生成文本，直到遇到结束标记
    while kwargs["stop"] not in generated_text:
        # 使用模型生成下一个词
        with torch.no_grad():
            output = model(input_ids)
            next_token_logits = output.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1)

        # 将生成的词添加到文本中
        generated_text += tokenizer.decode(next_token_id)

        # 更新输入，以便生成下一个词
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(1)], dim=-1)

        yield generated_text.lstrip("<sep>").rstrip(kwargs["stop"])
        # 控制生成的最大长度，以防止无限循环
        if input_ids.shape[1] >= kwargs["max_length"]:
            break


def generate_prompts(tokenizer, messages):
    # Call the function and get the result
    result = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=False,
    )

    return result


def load_chat_template(tokenizer, args):
    if args.chat_template is not None:
        try:
            with open(args.chat_template, "r") as f:
                chat_template = f.read()
        except OSError:
            # If opening a file fails, set chat template to be args to
            # ensure we decode so our escape are interpreted correctly
            chat_template = codecs.decode(args.chat_template, "unicode_escape")

        tokenizer.chat_template = chat_template
        logger.info(f"Chat template loaded from input: {args.chat_template}.")
    elif tokenizer.chat_template is not None:
        logger.info(f"Chat template loaded from tokenizer.")
    else:
        # throw a warning if no chat template is provided
        logger.warning("WARNING: No chat template provided. chat completion won't work.")


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
        "--embedding-path",
        type=str,
        default="BAAI/bge-large-zh-v1.5",
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
        default="0.0.0.0",
        help="host name"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8051,
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
    register_model_info(ModelCard(
        id=args.model_name,
    ))

    if args.cpu_only:
        device_map = 'cpu'
        device = torch.device("cpu")
    else:
        device_map = 'auto'
        device = torch.device("cuda")

    # Load LLM
    model, tokenizer = load_model_and_tokenizer(args.model_path, device_map)
    load_chat_template(tokenizer, args)

    # load Embedding
    embedding_model = SentenceTransformer(args.embedding_path, device="cuda")
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
