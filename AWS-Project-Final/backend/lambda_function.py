"""
Bep Thong Minh - AWS Lambda API
Runtime: Python 3.12
Handler: lambda_function.lambda_handler
DynamoDB table: MealRecipesTable (partition key RecipeID, String)
Supports API Gateway REST API v1 and HTTP API v2 events.
"""
import json
import logging
import os
import random
import re
import unicodedata
from decimal import Decimal

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

REGION = os.environ.get("AWS_REGION", "ap-southeast-1")
TABLE_NAME = os.environ.get("DYNAMODB_TABLE", "MealRecipesTable")
MAX_INPUT_LENGTH = 200
MAX_KEYWORDS = 10
dynamodb = boto3.resource("dynamodb", region_name=REGION)
table = dynamodb.Table(TABLE_NAME)
bedrock = boto3.client("bedrock-runtime", region_name=REGION)
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "apac.amazon.nova-lite-v1:0")

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type,Authorization",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    "Content-Type": "application/json; charset=utf-8",
}


def json_default(value):
    if isinstance(value, Decimal):
        return int(value) if value % 1 == 0 else float(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": CORS_HEADERS,
        "body": json.dumps(body, ensure_ascii=False, default=json_default),
    }


def normalize(value):
    text = str(value or "").strip().lower()
    text = text.replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def csv_values(value, field_name):
    if not value:
        return []
    if len(value) > MAX_INPUT_LENGTH:
        raise ValueError(f"Tham số {field_name} vượt quá {MAX_INPUT_LENGTH} ký tự.")
    values = [part.strip() for part in value.split(",") if part.strip()]
    if any(len(part) > 60 for part in values):
        raise ValueError(f"Mỗi giá trị trong {field_name} tối đa 60 ký tự.")
    return [v for v in values[:MAX_KEYWORDS] if not any(c in v for c in "<>{}$`\\;")]


def query_params(event):
    # HTTP API v2 may use rawQueryString and requestContext.http.method.
    params = event.get("queryStringParameters") or {}
    if params:
        return params
    raw = event.get("rawQueryString", "")
    from urllib.parse import parse_qs
    return {k: v[0] for k, v in parse_qs(raw).items() if v}


def get_all_recipes():
    items = []
    kwargs = {}
    while True:
        result = table.scan(**kwargs)
        items.extend(result.get("Items", []))
        last_key = result.get("LastEvaluatedKey")
        if not last_key:
            break
        kwargs = {"ExclusiveStartKey": last_key}
    logger.info("Loaded %s recipes from %s", len(items), TABLE_NAME)
    return items


def has_any(values, search_terms):
    normalized_values = [normalize(v) for v in values]
    for term in search_terms:
        t = normalize(term)
        if any(t in v or v in t for v in normalized_values if v and t):
            return True
    return False


def has_tag(recipe, field, tag):
    return tag in (recipe.get(field) or [])


def matches(recipe, filters):
    ingredients = recipe.get("Ingredients") or []
    allergens = recipe.get("Allergens") or []
    nutrition = recipe.get("Nutrition") or {}

    if filters["ingredients"] and not has_any(ingredients, filters["ingredients"]):
        return False
    if filters["health"] and not has_tag(recipe, "HealthTags", filters["health"]):
        return False
    if filters["need"] and not (
        has_tag(recipe, "HealthTags", filters["need"]) or
        has_tag(recipe, "DietTags", filters["need"])
    ):
        return False
    if filters["mood"] and not has_tag(recipe, "MoodTags", filters["mood"]):
        return False
    # Exclude a recipe if a requested allergen appears in either ingredient
    # names or the recipe's declared allergen list.
    if filters["allergies"] and (
        has_any(ingredients, filters["allergies"]) or
        has_any(allergens, filters["allergies"])
    ):
        return False
    if filters["avoid"] and has_any(ingredients, filters["avoid"]):
        return False

    numeric_rules = (
        ("maxTime", "CookingTimeMinutes"),
        ("maxBudget", "EstimatedCostVND"),
    )
    for param, field in numeric_rules:
        if filters[param] is not None:
            try:
                value = float(recipe.get(field))
            except (TypeError, ValueError):
                return False
            if value > filters[param]:
                return False

    for param, field in (("maxSodium", "SodiumMg"), ("maxCarbs", "CarbsG")):
        if filters[param] is not None:
            try:
                value = float(nutrition[field])
            except (KeyError, TypeError, ValueError):
                # Missing nutrition information is not treated as safe.
                return False
            if value > filters[param]:
                return False
    return True


def ingredient_score(recipe, keywords):
    ingredients = [normalize(v) for v in (recipe.get("Ingredients") or [])]
    return sum(1 for term in keywords if any(normalize(term) in ing for ing in ingredients))



def parse_chat_body(event):
    body = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        import base64
        body = base64.b64decode(body).decode("utf-8")
    try:
        payload = json.loads(body)
    except (TypeError, json.JSONDecodeError):
        raise ValueError("Nội dung gửi lên không phải JSON hợp lệ.")
    if not isinstance(payload, dict):
        raise ValueError("Dữ liệu trò chuyện không hợp lệ.")
    return payload


def chat_with_cooking_assistant(event):
    payload = parse_chat_body(event)
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        return response(400, {"error": "Hãy gửi ít nhất một tin nhắn."})
    if len(raw_messages) > 12:
        raw_messages = raw_messages[-12:]

    messages = []
    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = str(item.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        if len(content) > 1800:
            raise ValueError("Mỗi tin nhắn tối đa 1.800 ký tự.")
        messages.append({"role": role, "content": [{"text": content}]})

    if not messages or messages[-1]["role"] != "user":
        return response(400, {"error": "Tin nhắn cuối cùng phải là câu hỏi của người dùng."})

    context = payload.get("context") or {}
    ingredients = str(context.get("ingredients") or "").strip()[:500]
    context_note = ""
    if ingredients:
        context_note = "\\nNguyên liệu người dùng đang có trên trang: " + ingredients

    system_prompt = (
        "Bạn là Bếp Thông Minh, trợ lý AI chuyên hỗ trợ nấu ăn tại nhà. "
        "Luôn trả lời bằng tiếng Việt tự nhiên, dễ hiểu, thân thiện và thực tế. "
        "Ưu tiên món ăn Việt Nam và nguyên liệu dễ mua. Bạn có thể: gợi ý món từ nguyên liệu; "
        "hướng dẫn công thức theo từng bước, định lượng và thời gian; đề xuất nguyên liệu thay thế; "
        "lập danh sách đi chợ; giải thích lỗi thường gặp khi nấu; đưa ra ước tính dinh dưỡng khi được hỏi. "
        "Khi hướng dẫn nấu thịt, cá, trứng hoặc hải sản, nhắc người dùng nấu chín an toàn khi phù hợp. "
        "Không bịa rằng một cửa hàng đang có hàng hoặc một nguyên liệu có sẵn ở địa điểm cụ thể. "
        "Không đưa chẩn đoán hay phác đồ điều trị; thông tin dinh dưỡng chỉ mang tính tham khảo. "
        "Nếu thiếu thông tin quan trọng, hỏi lại ngắn gọn. Trình bày câu trả lời bằng các bước hoặc gạch đầu dòng khi hữu ích."
        + context_note
    )

    try:
        result = bedrock.converse(
            modelId=BEDROCK_MODEL_ID,
            system=[{"text": system_prompt}],
            messages=messages,
            inferenceConfig={"maxTokens": 700, "temperature": 0.5},
        )
        content_blocks = result.get("output", {}).get("message", {}).get("content", [])
        answer = "\\n".join(block.get("text", "") for block in content_blocks if block.get("text"))
        if not answer:
            raise RuntimeError("Mô hình không trả về nội dung.")
        return response(200, {"reply": answer, "model": BEDROCK_MODEL_ID})
    except ClientError:
        logger.exception("Amazon Bedrock request failed")
        return response(502, {"error": "Chatbot chưa gọi được Amazon Bedrock. Kiểm tra quyền IAM, model access và BEDROCK_MODEL_ID."})
    except Exception:
        logger.exception("Cooking assistant failed")
        return response(500, {"error": "Chatbot gặp lỗi nội bộ. Hãy thử lại sau."})


def lambda_handler(event, context):
    method = (
        event.get("httpMethod")
        or (event.get("requestContext", {}).get("http") or {}).get("method")
        or "GET"
    ).upper()
    path = (event.get("path") or event.get("rawPath") or event.get("resource") or "").rstrip("/")

    if method == "OPTIONS":
        return response(200, {"message": "CORS OK"})

    # POST /recipes is reserved for the cooking assistant; no food-ordering routes remain.
    if method == "POST" and (path.endswith("/recipes") or path.endswith("/chat")):
        try:
            return chat_with_cooking_assistant(event)
        except ValueError as exc:
            return response(400, {"error": str(exc)})
        except Exception:
            logger.exception("Unhandled cooking assistant error")
            return response(500, {"error": "Lỗi khi xử lý câu hỏi nấu ăn."})

    if method != "GET":
        return response(405, {"error": "Chỉ hỗ trợ GET để tìm công thức và POST để trò chuyện với trợ lý nấu ăn."})

    try:
        params = query_params(event)
        filters = {
            "ingredients": csv_values(params.get("ingredients", ""), "ingredients"),
            "allergies": csv_values(params.get("allergies", ""), "allergies"),
            "avoid": csv_values(params.get("avoid", ""), "avoid"),
            "health": (params.get("health") or "").strip(),
            "need": (params.get("need") or "").strip(),
            "mood": (params.get("mood") or "").strip(),
            "maxTime": None,
            "maxBudget": None,
            "maxSodium": None,
            "maxCarbs": None,
        }
        allowed_tags = {
            "health": {"low_sodium", "controlled_carb", "low_saturated_fat", "light", "high_protein"},
            "need": {"high_protein", "vegetarian", "vegan", "low_calorie"},
            "mood": {"comforting", "fresh", "energizing", "light"},
        }
        for key, allowed in allowed_tags.items():
            if filters[key] and filters[key] not in allowed:
                return response(400, {"error": f"Giá trị bộ lọc {key} không hợp lệ."})

        numeric_limits = {
            "maxTime": (600, 0),
            "maxBudget": (10000000, 0),
            "maxSodium": (100000, 0),
            "maxCarbs": (1000, 0),
        }
        for key, (upper, lower) in numeric_limits.items():
            raw = params.get(key)
            if raw not in (None, ""):
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    return response(400, {"error": f"Tham số {key} phải là số."})
                if value < lower or value > upper:
                    return response(400, {"error": f"Tham số {key} nằm ngoài giới hạn cho phép."})
                filters[key] = value

        logger.info("RequestId=%s filters=%s random=%s",
                    getattr(context, "aws_request_id", "local"), filters,
                    params.get("random") == "true")
        recipes = get_all_recipes()
        matched = [item for item in recipes if matches(item, filters)]
        matched.sort(key=lambda item: ingredient_score(item, filters["ingredients"]), reverse=True)
        if params.get("random", "false").lower() == "true":
            random.shuffle(matched)
            matched = matched[:6]

        return response(200, {
            "recipes": matched,
            "count": len(matched),
            "filters": filters,
            "source": "dynamodb",
        })
    except ValueError as exc:
        return response(400, {"error": str(exc)})
    except ClientError:
        logger.exception("DynamoDB request failed")
        return response(500, {"error": "Không truy cập được DynamoDB. Kiểm tra tên bảng và IAM Role."})
    except Exception:
        logger.exception("Unhandled Lambda error")
        return response(500, {"error": "Lỗi nội bộ. Xem CloudWatch Logs để biết chi tiết."})
