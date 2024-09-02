from ml_pipeline import MLPipeline
import itertools
from pathlib import Path
import os

class MLTester:
    scalers = ["MinMaxScaler", "StandardScaler"]
    pca_choices = [None, "Number of components", "Percentage of Variance"]

    def __init__(self, choice:MLPipeline.Data, csv:Path) -> None:
        self.testdocu = csv
        self.data = MLPipeline(choice)
        self.pca_args = {None:[None],
                         "Percentage of Variance":[25,50,75,90],
                         "Number of components": self.calc_pca_components()
                        }
        self.test_fn()

    def calc_pca_components(self):
        """calculate a list of possible numbers of pca components."""
        # max number of components:
        n = len(self.data.feature_names)
        # step:
        m = n // 10 + 1
        return list(range(2,n,m))

    def test_fn(self):
        """Cycle through the possible variants
        for one of the given datasets."""
        pre_list = [(None,None,None)] + [
            (sc,pca,arg) for sc in self.scalers
                         for pca in self.pca_choices
                         for arg in self.pca_args[pca]]
        iteration_list = itertools.product(self.data.ModelNames, pre_list)
        
        for name, (sc, pca, arg) in iteration_list:
            self.data.X = self.data.X_origin.copy()
            self.data.y = self.data.y_origin.copy()

            worked = self.data.preprocess_data(sc,pca,arg)
            if not worked:
                self.data.documentation_to_csv("PCA not possible",
                                               self.testdocu)
                continue
            
            self.data.create_fit_model(name)
            ev = self.data.evaluate()
            self.data.documentation_to_csv(ev,self.testdocu)


result_file = Path(__file__).parent / "Evaluation/model_testing.csv"
for dataset in MLPipeline.Datasets:
    MLTester(dataset, csv=result_file)
print(f"\n" + (""*60),
      "\nTest finished. You can find the results in:\n{result_file}")
