from agent.model_generation.model_generator import ModelGenerator
from representation.task import Operator


class EffectOnlyModelGenerator(ModelGenerator):
    def __init__(self):
        pass

    def _generate(self, forward=None, backward=None, execution=None):
        operators = []
        for idx, bc in enumerate(backward):
            add_effects, del_effects = bc
            params = self._get_params(add_effects | del_effects)
            name = "(op_" + str(idx)
            for f in params:
                name += " " + f
            name += ")"
            op = Operator(name, pos_preconds=set(), neg_preconds=set(), default_static=False, statics=set(),
                          add_effects=add_effects, del_effects=del_effects)
            operators.append(op)
        return operators

    def _get_params(self, fluents):
        """
        returns a list containing the parameters mentioned in the fluents

        """
        objects_all = set()
        for fluent in fluents:
            objects = fluent.replace("(","").replace(")","").split(" ")[1:]
            objects_all.update(objects)

        return objects_all

