from utils.model import perceptron
from utils.all_utils import prepare_data, save_model, load_model, save_plot
import pandas as pd
import os
import logging
import numpy as np

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename= os.path.join(log_dir,"running_logs_or.log"),level=logging.INFO, format=logging_str, filemode="a")

def main(df,plotName, eta, epochs):
    ETA = eta
    EPOCHS = epochs
    x,y = prepare_data(df)
    model = perceptron(eta= ETA,epochs= EPOCHS)
    model.fit(x, y)
    save_model(model,"or_model")
    #load model then predict
    saved_model = load_model("or_model")
    input = np.array([[0,0]])
    print(saved_model.predict(x=input))
    save_plot(df, plotName, model)


if __name__ == '__main__':
    OR = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,1,1,1]
    }
    ETA = 0.3
    EPOCHS = 10
    df = pd.DataFrame(OR)
    try:
        main(df,plotName="or.png", eta=ETA,epochs=EPOCHS)
    except Exception as e:
        logging.exception(f"Exception occured : {e}")
        raise e
    