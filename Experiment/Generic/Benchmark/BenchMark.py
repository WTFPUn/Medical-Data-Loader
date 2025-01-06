import os
import json
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, List, Dict, Type
import time
from SimpleITK import GetImageFromArray, WriteImage
import numpy as np
from scipy.special import softmax

from ...Generic import ModelTrainer, Metric
from ...DataEngine import DataEngine
from ...DataEngine.types import (
    DatasetMetaData,
    DatasetConfig,
    SimpleDatasetMetaData,
    KFOLDDatasetMetaData,
)
from ...DataEngine import get_array_by_subplot, VisualBlock, SubplotBlock

T, U = TypeVar("T"), TypeVar("U")


@dataclass
class ModelTrainerInit(Generic[T, U]):
    model: Type[ModelTrainer[T, U]]
    model_name: str
    experiment_name: str
    model_path: str
    epochs: int


class BenchmarkGeneric(ABC, Generic[T, U]):
    def __init__(
        self,
        bench_mark_name: str,
        meta_data_path: str,
        dataset_config: DatasetConfig,
        metrics: List[Metric[T, U]],
        num_classes: int,
        logger: logging.Logger,
    ):
        self.logger = logger.getChild(self.__class__.__name__)
        self.bench_mark_name = bench_mark_name
        self.meta_data_path = meta_data_path
        self.dataset_config = dataset_config
        self.device = dataset_config.device
        self.num_classes = num_classes
        self.metrics = metrics
        self.device = dataset_config.device

        try:
            with open(meta_data_path, "r", encoding="utf-8") as f:
                self.meta_data = SimpleDatasetMetaData.model_validate(json.load(f))
        except Exception as e:
            self.logger.error(
                "Failed to load meta data from %s with error: %s",
                meta_data_path,
                e,
                extra={"contexts": "load meta data"},
            )
            return

        self.data_engine = DataEngine(
            num_classes, self.meta_data, dataset_config, logger
        )
        self.train_method: List[ModelTrainer[T, U]] = []

        self.logger.info(
            "Benchmark initialized.", extra={"contexts": "initialize benchmark"}
        )

    def add_trainer(self, model_trainer: ModelTrainerInit[T, U], **kwargs) -> None:
        self.train_method.append(
            model_trainer.model(
                logger=self.logger.getChild(model_trainer.model_name),
                losses=[],
                metrics=self.metrics,
                device=self.device,
                load_model_path=model_trainer.model_path,
                num_classes=self.num_classes,
                name=model_trainer.model_name,
                **kwargs,
            )
        )

        self.logger.info(
            "Added trainer %s",
            model_trainer.model_name,
            extra={"contexts": "add trainer"},
        )

    def run(self, save_result: bool = True) -> None:
        """
        a method to run the benchmark

        Parameters:
        save_result (bool): whether to save the result or not(result can be image or anything else. You can custom )

        """

        os.makedirs("BenchMark", exist_ok=True)

        results = {}

        for trainer in self.train_method:
            test_dataset = self.data_engine.get_dataloader("test", 1, False, 0)

            result = trainer.test(test_dataset)
            # return result eample
            # to_return = {
            #     "metrics": {},
            #     # output is the returned value from the model
            #     "output": [],
            #     "ground_truth": []
            # }
            self.logger.info(
                "Trainer %s finished testing", trainer.name, extra={"contexts": "run"}
            )
            results[trainer.name] = result

        if save_result:
            self.save_result(results)

    def save_result(self, result):
        """
        # TODO: will write after complete the run method

        """

        # reformat from Dict[Model name, result] to each image

        # example of result_perbatch
        """
        [
            {
                "ground_truth": ground_truth of image 1,
                "output": [
                    {
                        "model_name": model 1,
                        "metrics": metrics of the model of image 1,
                        "output": output of the model of image 1,
                        "infer_time": time of inference
                        
                    },
                    {
                        "model_name": model 2,
                        "metrics": metrics of the model of image 1,
                        "output": output of the model of image 1,
                        "infer_time": time of inference
                    }
                ]
            },
            {
                "ground_truth": ground_truth of image 2,
                "output": [
                    {
                        "model_name": model 1,
                        "metrics": metrics of the model of image 2,
                        "output": output of the model of image 1,
                        "infer_time": time of inference
                    },
                    {
                        "model_name": model 2,
                        "metrics": metrics of the model of image 2,
                        "output": output of the model of image 1,
                        
                        "infer_time": time of inference
                    }
                ]
            }
        ]
        """
        # path to save the result
        path = f"BenchMark/{time.time()}"
        os.makedirs(path, exist_ok=True)

        result_perbatch = []
        for key in result.keys():
            for i in range(len(result[key]["output"])):
                if i >= len(result_perbatch):
                    result_perbatch.append(
                        {
                            "ground_truth": result[key]["ground_truth"][i],
                            "input": np.squeeze(np.squeeze(result[key]["input"][i])),
                            "output": [],
                        }
                    )
                output = result[key]["output"][i]

                # make it from (1, C, H, W, D) to (C, H, W, D)
                output = np.squeeze(output)

                # make it from (C, H, W, D) to (1, H, W, D) by argmax
                output = np.argmax(output, axis=0)

                result_perbatch[i]["output"].append(
                    {
                        "model_name": key,
                        "metrics": result[key]["metrics"],
                        "output": output,
                        "infer_time": result[key]["infer_time"][i],
                    }
                )

        print(result_perbatch)
        self.generate_input(result_perbatch, path)
        self.generate_ground_truth(result_perbatch, path)
        self.generate_nii(result_perbatch, path)
        self.generate_report(result_perbatch, path)

    def generate_report(self, result_perbatch, path):
        # write csv file to generate report
        import pandas as pd

        # column head: image, model, metric_name, ..., infer_time
        # row: image 1, model 1, metric 1, ..., infer_time
        #      image 1, model 1, metric 2, ..., infer_time

        # create a list of dict to store the data
        data = []
        for i in range(len(result_perbatch)):
            for model_out in result_perbatch[i]["output"]:
                data.append(
                    {
                        "image": i,
                        "model": model_out["model_name"],
                        "infer_time": model_out["infer_time"],
                    }
                )

                for metric_name, metric_value in model_out["metrics"].items():
                    data[-1][metric_name] = metric_value[i]

        df = pd.DataFrame(data)
        df.to_csv(path + "/report.csv")

    def generate_input(self, result_perbatch, path):
        # generate input from the output
        os.makedirs(path + f"/input", exist_ok=True)
        for i in range(len(result_perbatch)):
            img = GetImageFromArray(result_perbatch[i]["input"])
            WriteImage(img, path + f"/input/{i}.nii.gz")

    def generate_video(self, result_perbatch, path):
        # generate video from the output
        # write SubplotBlock to generate video layout is 1(ground truth) + number of model for each image
        from matplotlib.colors import ListedColormap

        cmap = ListedColormap(["black", "green", "blue"])

        # for i in range(len(result_perbatch)):

    def generate_ground_truth(self, result_perbatch, path):
        # generate ground truth from the output
        os.makedirs(path + f"/ground_truth", exist_ok=True)
        for i in range(len(result_perbatch)):
            img = GetImageFromArray(result_perbatch[i]["ground_truth"])
            WriteImage(img, path + f"/ground_truth/{i}.nii.gz")

    def generate_nii(self, result_perbatch, path):
        # generate nii file from the output

        for i in range(len(result_perbatch)):
            os.makedirs(path + f"/output/{i}/nii", exist_ok=True)

            for model_out in result_perbatch[i]["output"]:
                img = GetImageFromArray(model_out["output"])
                WriteImage(
                    img, path + f"/output/{i}/nii/{model_out['model_name']}.nii.gz"
                )
