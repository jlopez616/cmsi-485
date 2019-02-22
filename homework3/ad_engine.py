import itertools
import unittest
import math
import numpy as np
from pomegranate import *
import sys

class AdEngine:

    def __init__(self, data_file, structure, dec_vars, util_map):
        """
        Responsible for initializing the Decision Network of the
        AdEngine from the structure discovered by Tetrad

        #Pomegrante

        :param string data_file: path to csv file containing data on which
        the network's parameters are to be learned
        :param tuple structure: tuple of tuples specifying parental
        relationships between variables in the network; see Pomegranate docs
        for the expected format. Example:
          ((), (0), (1)) represents nodes: [0] -> [1] -> [2]
        :param list dec_vars: list of string names of variables to be
        considered decision points for the agent. Example:
          ["Ad1", "Ad2"]
        :param dict util_map: discrete, tabular, utility map whose keys
        are variables in network that are parents of a utility node, and
        values are dictionaries mapping that variable's values to a utility
        score, e.g.
          {
            "X": {0: 20, 1: -10}
          }
        represents a utility node with single parent X whose value of 0
        has a utility score of 20, and value 1 has a utility score of -10
        """
        self.data_file = data_file;
        self.text = np.genfromtxt(data_file, names = True, dtype = None, delimiter=',', encoding = None);
        self.names = self.text.dtype.names
        self.structure = structure
        self.dec_vars = dec_vars;
        self.util_map = util_map;

        self.dec_var_combos = [list(np.unique(self.text[idk])) for idk in dec_vars]

        self.bn = BayesianNetwork.from_structure(self.text.view((int, len(self.names))), self.structure, state_names = self.names);
        self.query = list(self.util_map)[0]
        self.query_index = self.names.index(self.query) #index of S in our list




    def decide(self, evidence):
        """
        Given some observed demographic "evidence" about a potential
        consumer, selects the ad content that maximizes expected utility
        and returns a dictionary over any decision variables and their
        best values

        :param dict evidence: dict mapping network variables to their
        observed values, of the format: {"Obs1": val1, "Obs2": val2, ...}
        :return: dict of format: {"DecVar1": val1, "DecVar2": val2, ...}
        """
        best_combo, best_util = {}, -math.inf

        probList = self.bn.predict_proba(evidence)
        probVals = probList[self.query_index].parameters[0].values()

        for element in itertools.product(*self.dec_var_combos):
            newDict = dict(zip(self.dec_vars, list(element))) #conforms to style for predict proba
            newDict.update(evidence)
            cpt = self.bn.predict_proba(newDict)[self.query_index].parameters[0]
            highestEU = 0;
            for table in range(len(cpt)):
                highestEU += self.util_map[self.query][table] * cpt[table]
            if highestEU > best_util:
                best_util = highestEU
                best_combo = dict(zip(self.dec_vars, list(element)))
        return best_combo


class AdEngineTests(unittest.TestCase):
   def test_defendotron_ad_engine_t1(self):
       engine = AdEngine(
           data_file = 'hw3_data.csv',
           dec_vars = ["Ad1", "Ad2"],
           structure = ((),(),(0, 9),(6,),(0, 1),(1, 8),(),(2, 5),(),()),
           util_map = {
               "S": {0: 0, 1: 5000, 2: 17660}
           }
       )
       self.assertEqual(engine.decide({"T": 1}), {"Ad1": 0, "Ad2": 1})
       self.assertIn(engine.decide({"F": 1}), [{"Ad1": 1, "Ad2": 0},{"Ad1": 1, "Ad2": 1}])
       self.assertEqual(engine.decide({"G": 1, "T": 0}), {"Ad1": 1, "Ad2": 1})

   def test_defendotron_ad_engine_t2(self):
       engine = AdEngine(
           data_file = 'hw3_data.csv',
           dec_vars = ["Ad1"],
           structure = ((),(),(0, 9),(6,),(0, 1),(1, 8),(),(2, 5),(),()),
           util_map = {
               "S": {0: 0, 1: 5000, 2: 17660},
           }
       )
       self.assertEqual(engine.decide({"A": 1}), {"Ad1": 0})
       self.assertEqual(engine.decide({"P": 1, "A": 0}), {"Ad1": 1})
       self.assertIn(engine.decide({"A": 1, "G": 0, "T": 1}), [{"Ad1": 0}, {"Ad1": 1}])


if __name__ == "__main__":
  unittest.main()
