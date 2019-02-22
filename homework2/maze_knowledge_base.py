'''
maze_knowledge_base.py

Specifies a simple, Conjunctive Normal Form Propositional
Logic Knowledge Base for use in Grid Maze pathfinding problems
with side-information.
'''
from maze_clause import MazeClause
import unittest
import itertools

class MazeKnowledgeBase:

    def __init__ (self):
        self.clauses = set();

    def tell (self, clause):
        """
        Adds the given clause to the CNF MazeKnowledgeBase
        Note: we expect that no clause added this way will ever
        make the KB inconsistent (you need not check for this)
        """
        self.clauses.add(clause);

    def ask (self, query):
        """
        Given a MazeClause query, returns True if the KB entails
        the query, False otherwise
        """
        currentClauses = set(self.clauses);
        newClauses = set();

        for cell, bool in query.props.items():
            currentClauses.add(MazeClause([(cell, not bool)]));

        while True:
            iterations = list(itertools.combinations(currentClauses, 2));
            for pair in iterations:
                toResolve = MazeClause.resolve(pair[0], pair[1]);

                for clauses in toResolve:
                    if clauses.is_empty():
                        return True;
                newClauses = newClauses | toResolve;

            if newClauses <= currentClauses:
                return False;

            currentClauses = currentClauses | newClauses;


class MazeKnowledgeBaseTests(unittest.TestCase):
    def test_mazekb1(self):
        kb = MazeKnowledgeBase()
        kb.tell(MazeClause([(("X", (1, 1)), True)]))
        self.assertTrue(kb.ask(MazeClause([(("X", (1, 1)), True)])))

    def test_mazekb2(self):
        kb = MazeKnowledgeBase()
        kb.tell(MazeClause([(("X", (1, 1)), False)]))
        kb.tell(MazeClause([(("X", (1, 1)), True), (("Y", (1, 1)), True)]))
        self.assertTrue(kb.ask(MazeClause([(("Y", (1, 1)), True)])))

    def test_mazekb3(self):
        kb = MazeKnowledgeBase()
        kb.tell(MazeClause([(("X", (1, 1)), False), (("Y", (1, 1)), True)]))
        kb.tell(MazeClause([(("Y", (1, 1)), False), (("Z", (1, 1)), True)]))
        kb.tell(MazeClause([(("W", (1, 1)), True), (("Z", (1, 1)), False)]))
        kb.tell(MazeClause([(("X", (1, 1)), True)]))
        self.assertTrue(kb.ask(MazeClause([(("W", (1, 1)), True)])))
        self.assertFalse(kb.ask(MazeClause([(("Y", (1, 1)), False)])))

    def test_mazekb4(self):
        kb = MazeKnowledgeBase()
        kb.tell(MazeClause([(("X", (1, 1)), False), (("Y", (1, 1)), True), (("W", (1, 1)), True)]))
        kb.tell(MazeClause([(("W", (1, 1)), False), (("Z", (1, 1)), False), (("S", (1, 1)), True)]))
        kb.tell(MazeClause([(("S", (1, 1)), False), (("T", (1, 1)), False)]))
        kb.tell(MazeClause([(("X", (1, 1)), True), (("T", (1, 1)), True)]))
        kb.tell(MazeClause([(("W", (1, 1)), True)]))
        kb.tell(MazeClause([(("T", (1, 1)), True)]))
        self.assertTrue(kb.ask(MazeClause([(("Z", (1, 1)), False)])))

    def test_mazekb5(self):
        kb = MazeKnowledgeBase()
        kb.tell(MazeClause([(("X", (1, 1)), False), (("Y", (1, 1)), True), (("W", (1, 1)), True)]))
        kb.tell(MazeClause([(("W", (1, 1)), False), (("Z", (1, 1)), False), (("S", (1, 1)), True)]))
        kb.tell(MazeClause([(("S", (1, 1)), False), (("T", (1, 1)), False)]))
        kb.tell(MazeClause([(("X", (1, 1)), True), (("T", (1, 1)), True)]))
        kb.tell(MazeClause([(("W", (1, 1)), True)]))
        kb.tell(MazeClause([(("T", (1, 1)), True)]))
        self.assertTrue(kb.ask(MazeClause([(("Z", (1, 1)), True), (("W", (1, 1)), True)])))

    def test_mazekb6(self):
        kb = MazeKnowledgeBase()
        kb.tell(MazeClause([(("X", (1, 1)), False), (("Y", (1, 1)), False), (("Z", (1, 1)), False)]))
        kb.tell(MazeClause([(("X", (1, 1)), True)]))
        self.assertFalse(kb.ask(MazeClause([(("Z", (1, 1)), False)])))
        kb.tell(MazeClause([(("Y", (1, 1)), True)]))
        self.assertTrue(kb.ask(MazeClause([(("Z", (1, 1)), False)])))


if __name__ == "__main__":
    unittest.main()
