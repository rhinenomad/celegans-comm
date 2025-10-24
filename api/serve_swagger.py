# api/serve_swagger.py
import yaml, pathlib
from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import JSONResponse, PlainTextResponse

ROOT = pathlib.Path(__file__).parent
YAML_PATH = ROOT / "swagger.yaml"

# 1) 禁用内置 /openapi.json 与 /docs（我们自己提供）
app = FastAPI(title="Connectome API (spec loader)",
              docs_url=None, redoc_url=None, openapi_url=None)

# 2) 读取你的 swagger.yaml
spec_yaml_text = YAML_PATH.read_text(encoding="utf-8")
spec = yaml.safe_load(spec_yaml_text)

# 3) 提供自定义的 OpenAPI 输出
@app.get("/openapi.json", include_in_schema=False)
def openapi_json():
    return JSONResponse(content=spec)

@app.get("/openapi.yaml", include_in_schema=False)
def openapi_yaml():
    return PlainTextResponse(spec_yaml_text, media_type="application/yaml")

# 4) 提供 Swagger UI，指向我们自己的 /openapi.json
@app.get("/docs", include_in_schema=False)
def docs():
    return get_swagger_ui_html(openapi_url="/openapi.json", title="Swagger UI")

@app.get("/health")
def health():
    return {"status": "ok"}
