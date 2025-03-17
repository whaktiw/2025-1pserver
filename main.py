import uvicorn
from fastapi import FastAPI, Query

from fastapi.middleware.cors import CORSMiddleware
from irisModel import IrisMachineLearning, IrisSpecies

app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    COR,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = IrisMachineLearning()

@app.get("/")
async def root():
    return {"message": "Hello This is iris classfier"}

@app.get("/predict")
async def predict():
    pred, prob = model.predict_species(8,1,8,1)
    return {"predict" : pred,
            "probability" : prob}

@app.post("/predict")
async  def predict_species(iris: IrisSpecies):
    pred, prob = model.predict_species(
        iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width )
    #print(f'pred={prob}')
    return {"prediction": pred,
            "probability": prob}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)