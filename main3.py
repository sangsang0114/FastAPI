from fastapi import FastAPI

app = FastAPI()

fake_items_db = [
    {"item_name": "FOO"},
    {"item_name": "Bar"},
    {"item_name": "Baz"},
]


@app.get("/test/1")
async def read_item(skip: int = 0, limit: int = 10):
    print(skip, " ", limit)
    return fake_items_db[skip : skip + limit]


resultJson = {
    "item_name": "Foo",
    "num": 0,
}


@app.get("/test/2")
async def test(num: int = 0, text: str = "안녕!"):
    print(num, text)
    resultJson["num"] = num
    resultJson["item_name"] = text
    return resultJson
