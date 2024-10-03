from fastapi import FastAPI
import torch

app = FastAPI()

model = torch.load('models/model.pt')


@app.post("/")
def read_root(data: dict):
    size, weight = data['size'], data['weight']
    juiciness = (model(torch.tensor([size, weight]))[0].item())
    return {'juiciness': juiciness}
