from utils.model import perceptron
from utils.all_utils import prepare_data, save_model, load_model
import pandas as pd
import os
import logging
import numpy as np

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename= os.path.join(log_dir,"running_logs.log"),level=logging.INFO, format=logging_str, filemode="a")


AND = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,0,0,1]
}

df = pd.DataFrame(AND)
print(df)

x,y = prepare_data(df)
ETA = 0.3
EPOCHS = 100
model = perceptron(eta= ETA,epochs= EPOCHS)

model.fit(x,y)
save_model(model,"and_model")

#fetch model then predict
saved_model = load_model("and_model")
input = np.array([[0,0]])
print(saved_model.predict(x=input))
