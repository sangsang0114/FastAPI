# python -m uvicorn fastapi_lab:app --reload --port=8000
# '미니프로젝트2 참고용' 기반
from fastapi import FastAPI
from keras.models import load_model
from konlpy.tag import Okt
import option as opt
import data
import os

op = opt.Options()
okt = Okt()
tokenizer = data.get_tokenizer()
app = FastAPI()


def predict(news_nouns):
    model = ""
    if os.path.exists(op.model_o):
        model = load_model(op.model_o)
    result = model.predict(data.make_onehot(tokenizer, [news_nouns]))
    return result


@app.get("/news/predict")
async def read_item(news: str):
    news_nouns = okt.nouns(news)
    result = predict(news_nouns)
    return "증권 뉴스입니다." if result < 0.5 else "부동산 뉴스입니다."
