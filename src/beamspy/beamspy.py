import torch
from transformers.generation import LogitsProcessor

class _BeamSpyInternal:
    def __init__(self, tokenizer, print_recipes, print_beams):
        self._beamTree = {}
        self._beams = {}
        self._len = 0
        self._tokenizer = tokenizer
        self._print_recipes = print_recipes
        self._print_beams = print_beams
    def identifyBeams(self, input_ids):
        def _addPath(tree, path, tokPos, tokVal, beamId):
            cr = tree
            for e in path:
                if e not in cr:
                    cr[e] = {}
                cr = cr[e]
            cr[(tokPos, tokVal)] = beamId
        def _findBeam(beam, tree):
            if 1 == len(beam) and {} == tree:
                return [], 0
            retq = []
            cr = tree
            while True:
                found = False
                for k, v in cr.items():
                    tokPos, tokVal = k
                    if tokVal == beam[tokPos]:
                        found = True
                        retq.append((tokPos, tokVal))
                        break
                if not found:
                    print(beam, tree)
                    raise ValueError("Beam tree should contain prefix of new beam.")
                cr = cr[k]
                if not isinstance(cr, dict):
                    return retq, cr
            raise ValueError("New beam should match some previously stored beam.")
        def _collapseSingleChoices(tree):
            todo = [(None, None, tree)]
            while todo:
                parent, key, cr = todo.pop()
                toApp = []
                if 1 == len(cr):
                    k = list(cr.keys())[0]
                    v = cr[k]
                    if isinstance(v, dict):
                        cr.pop(k)
                        for j, vj in v.items():
                            cr[j] = vj
                        toApp = [[parent, key, cr]]
                    elif parent is not None:
                        parent[key] = v
                else:
                    toApp = [[cr, k, v] for k, v in cr.items() if isinstance(v, dict)]
                todo += toApp
            return tree
        beamlen = input_ids.shape[1]
        if 0 == beamlen:
            return [0]*input_ids.shape[0]
        retq = []
        newTree = {}
        for k, beam in enumerate(input_ids):
            path, parentBeamId = _findBeam(beam, self._beamTree)
            _addPath(newTree, path, beamlen-1, beam[-1], k)
            retq.append(parentBeamId)
        self._beamTree = _collapseSingleChoices(newTree)
        return retq
    def update(self, input_ids, batch_id, num_beams):
        input_ids = input_ids[batch_id*num_beams:(batch_id*num_beams+num_beams)]
        beam_len = input_ids.shape[1]
        if 0 == beam_len:
            self._beams = {k: [None, None] for k in range(num_beams)}
        else:
            self._len += 1
            beam_parents = self.identifyBeams(input_ids)
            self._beams = {k: [self._beams.get(parent), input_ids[k,-1].tolist()] for k, parent in enumerate(beam_parents)}
        recipes = [(p, input_ids[k,-1].tolist()) for k, p in enumerate(beam_parents)]
        if self._print_beams or self._print_recipes:
            print("----")
            for k, recipe in enumerate(recipes):
                beam_parent_idx, token = recipe
                aux = ""
                if self._print_recipes:
                    aux += "%d + %d" % (beam_parent_idx, token)
                if self._print_beams:
                    if self._print_recipes:
                        aux += ": "
                    aux += "%s" % str(self._followBeam(self._tokenizer.decode(self._beams[k])))
                print(aux)                
        return recipes
    def _followBeam(self, beam):
        retq = []
        while beam is not None:
            beam, token = beam
            if token is not None:
                 retq.append(token)
        return list(reversed(retq))
    def getBeams(self):
        return [self._followBeam(self._beams[k]) for k in range(len(self._beams))]

        
class BeamSpy(LogitsProcessor):
    def __init__(self, batch_size, tokenizer, print_recipes=False, print_beams=False):
        super().__init__()
        self._batch_size = textCount
        self._tokenizer = tokenizer
        self._print_beams = print_beams
        self._print_recipes = print_recipes or self._print_beams
        self._beam_spies = [_BeamSpyInternal(self._tokenizer, self._print_recipes, self._print_beams) for k in range(self._batch_size)]
    def _doWork(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, recipes) -> torch.FloatTensor:
        """
        BeamSpy._doWork: overload this function to implement your prefix-dependent logits processor
        
        Input arguments:
            input_ids: torch tensor of shape (batch_len*beam_count, beam_len), elements are the tokens already present in each beam.
            scores: torch tensor of shape (batch_len*beam_count, token_count), elements are the logits for the next tokens for each beam.
            recipes: list of lists of pairs of form (beam_parent_idx, token).
        Output arguments:
            scores: torch tensor of shape (batch_len*beam_count, token_count), elements are the updated logits.
        
        The recipes parameter should be interpreted as follows:
            - the recipes list has an element for each text in the input batch.
            - therefore, for each text in the input batch there is a list of recipes.
            - for a text in the batch, if the j-th recipe has beam_parent_idx i and token x, it means that the currently j-th likely beam was obtained by appending x to what was previously the i-th likely beam.
        
        The recommendation is for your function to keep track of the work it did for the current 0..m-th likeliest beams, and upon receiving recipes for beam updates, use the beam_parent_idx to select which cached work to reuse and for which beam.
        """
        ## TODO: overload this function to have it do something useful for you.
        return scores
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        num_beams = input_ids.shape[0] // self._batch_size
        recipes = [beam_spy.update(input_ids, text_id, num_beams) for text_id, beam_spy in enumerate(self._beam_spies)]
        return self._doWork(input_ids, scores, recipes)
