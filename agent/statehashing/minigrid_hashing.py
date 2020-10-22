from agent.statehashing.statehashing import StateHasher


class MinigridStateHasher(StateHasher):
    def __init__(self, detector):
        super().__init__()
        types_to_objs = detector.get_objects(inherit=True)
        self.relevant_objects = list(types_to_objs["graspable"]) \
                                + list(types_to_objs["agent"]) + \
                                list(types_to_objs["door"])
        self.relevant_objects.sort()

    """
    Note that this isn't the same thing as the internal state hashing, in that it doesn't return a hashcode, but rather
    a smaller representation that uniquely identifies a given state based on the positions of all the objects that can
    be in unique locations or states and takes up as little memory as possible.
    """
    def _hash(self, obs):
        # This should make the size of the Q tables substantially smaller and reduce the likelihood of memory errors,
        # but if needed we can shrink this down even smaller.
        substate = []
        for obj_name in self.relevant_objects:
            substate.append(obs['objects'][obj_name]['x'])
            substate.append(obs['objects'][obj_name]['y'])
            substate.append(obs['objects'][obj_name]['encoding'][2])
        return tuple(substate)
