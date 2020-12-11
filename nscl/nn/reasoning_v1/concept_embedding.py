
import torch
import torch.nn as nn
import torch.nn.functional as F
import jactorch
import jactorch.nn as jacnn

from jacinle.utils.cache import cached_property

__all__ = [
    'AttributeBlock', 'ConceptBlock', 'ConceptEmbedding'
]


"""QAS mode: if True, when judging if two objects are of same color, consider all concepts belongs to `color`."""
_query_assisted_same = False


def set_query_assisted_same(value):
    """Set the QAS mode."""
    global _query_assisted_same
    _query_assisted_same = value


class AttributeBlock(nn.Module):
    """Attribute as a neural operator."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.map = jacnn.LinearLayer(input_dim, output_dim, activation=None)


class ConceptBlock(nn.Module):
    """
    Concept as an embedding in the corresponding attribute space.
    """
    def __init__(self, embedding_dim, nr_attributes, threshold_normalize=None,attribute_agnostic=False,bilinear=False,coord_semantics=False):
        """
        Args:
            embedding_dim (int): dimension of the embedding.
            nr_attributes (int): number of known attributes.
            attribute_agnostic (bool): if the embedding in different embedding spaces are shared or not.
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.nr_attributes = nr_attributes
        self.attribute_agnostic = attribute_agnostic

        if bilinear:
            self.embedding = nn.Parameter(torch.randn(embedding_dim,embedding_dim))
        elif coord_semantics:
            self.embedding = nn.Parameter(torch.randn(4))
        else:
            if self.attribute_agnostic:
                self.embedding = nn.Parameter(torch.randn(embedding_dim))
            else:
                self.embedding = nn.Parameter(torch.randn(nr_attributes, embedding_dim))
        self.belong = nn.Parameter(torch.randn(nr_attributes) * 0.1)

        self.known_belong = False

        self.threshold_normalize = threshold_normalize

    def set_belong(self, belong_id):
        """
        Set the attribute that this concept belongs to.
        Args:
            belong_id (int): the id of the attribute.
        """
        self.belong.data.fill_(-100)
        self.belong.data[belong_id] = 100
        self.belong.requires_grad = False
        self.known_belong = True

    @property
    def normalized_embedding(self):
        """L2-normalized embedding in all spaces."""
        embedding = self.embedding / self.embedding.norm(2, dim=-1, keepdim=True)
        if self.attribute_agnostic:
            return jactorch.broadcast(embedding.unsqueeze(0), 0, self.nr_attributes)
        return embedding

    @property
    def log_normalized_belong(self):
        """Log-softmax-normalized belong vector."""
        return F.log_softmax(self.belong, dim=-1)

    @property
    def normalized_belong(self):
        """Softmax-normalized belong vector."""
        return F.softmax(self.belong, dim=-1)


class ConceptEmbedding(nn.Module):
    def __init__(self, attribute_agnostic,bilinear_relation,coord_semantics, threshold_normalize):
        super().__init__()

        self.attribute_agnostic = attribute_agnostic
        self.all_attributes = list()
        self.all_concepts = list()
        self.attribute_operators = nn.Module()
        self.concept_embeddings = nn.Module()

        self.margin = nn.Parameter(torch.tensor(0.0, requires_grad=False))
        self.tau = nn.Parameter(torch.tensor(0.1, requires_grad=False))

        self.bilinear_relation = bilinear_relation
        self.coord_semantics = coord_semantics
        self.threshold_normalize = threshold_normalize

    @property
    def nr_attributes(self):
        return len(self.all_attributes)

    @property
    def nr_concepts(self):
        return len(self.all_concepts)

    @cached_property
    def attribute2id(self):
        return {a: i for i, a in enumerate(self.all_attributes)}

    def init_attribute(self, identifier, input_dim, output_dim):
        assert self.nr_concepts == 0, 'Can not register attributes after having registered any concepts.'
        self.attribute_operators.add_module('attribute_' + identifier, AttributeBlock(input_dim, output_dim))
        self.all_attributes.append(identifier)
        # TODO(Jiayuan Mao @ 11/08): remove this sorting...
        self.all_attributes.sort()

    def init_concept(self, identifier, input_dim, known_belong=None):
        block = ConceptBlock(input_dim, self.nr_attributes, attribute_agnostic=self.attribute_agnostic,bilinear=self.bilinear_relation,coord_semantics=self.coord_semantics,threshold_normalize=self.threshold_normalize)
        self.concept_embeddings.add_module('concept_' + identifier, block)
        if known_belong is not None:
            block.set_belong(self.attribute2id[known_belong])
        self.all_concepts.append(identifier)

    def get_belongs(self):
        belongs = dict()
        for k, v in self.concept_embeddings.named_children():
            belongs[k] = self.all_attributes[v.belong.argmax(-1).item()]
        class_based = dict()
        for k, v in belongs.items():
            class_based.setdefault(v, list()).append(k)
        class_based = {k: sorted(v) for k, v in class_based.items()}
        return class_based

    def get_attribute(self, identifier):
        x = getattr(self.attribute_operators, 'attribute_' + identifier)
        return x.map

    def get_all_attributes(self):
        return [self.get_attribute(a) for a in self.all_attributes]

    def get_concept(self, identifier):
        return getattr(self.concept_embeddings, 'concept_' + identifier)

    def get_all_concepts(self):
        return {c: self.get_concept(c) for c in self.all_concepts}

    def get_concepts_by_attribute(self, identifier):
        return self.get_attribute(identifier), self.get_all_concepts(), self.attribute2id[identifier]

    #_margin = 0
    _margin_cross = 0.5
    #_tau = 0.1

    def similarity(self, query, identifier,k=1,mutual_exclusive=True, logit_semantics=False):
        #identifier is a concept
        #returns a list of log probabilities: prob that each object is the concept
        if k==1 or self.bilinear_relation:
            query = query[1]
        elif self.coord_semantics:
            query = query[3]
        else:
            query = query[2]

        attributes = self.all_attributes
        concept = self.get_concept(identifier)
        attribute_index = concept.belong.argmax(-1).item()

        if k==2: #two-place relation
            

            attr_identifier = attributes[attribute_index]
            mapping = self.get_attribute(attr_identifier)
            attr_id = self.attribute2id[attr_identifier]
            


            if self.bilinear_relation:
                query = mapping(query)
                logits = torch.einsum('ax,xy,by->ab',query,concept.embedding,query)
                log_prob = nn.LogSigmoid()(logits)
            else:
                num_objs = query.size(0)
                query = query.reshape(-1,query.size(-1))
                if not self.coord_semantics:
                    query = mapping(query)
                    concept_embedding = concept.normalized_embedding[attr_id]
                    #if not self.threshold_normalize:
                    query = query / query.norm(2, dim=-1, keepdim=True)
                    #else:
                    #    query_norm = query.norm(2, dim=-1, keepdim=True)
                    #    query = torch.where(query_norm<self.threshold_normalize, query, query / query_norm)
                else:
                    concept_embedding = concept.normalized_embedding

                
                logits = (torch.matmul(query,concept_embedding)/self.tau)
                if logit_semantics:
                    log_prob = logits
                else:
                    log_prob = nn.LogSigmoid()(logits)

                log_prob = log_prob.reshape(num_objs,num_objs)


        

        elif k==1:
            if logit_semantics:
                attr_identifier = attributes[attribute_index]
                mapping = self.get_attribute(attr_identifier)
                attr_id = self.attribute2id[attr_identifier]
            
                query = mapping(query)
                query = query / query.norm(2, dim=-1, keepdim=True)
                concept_embedding = concept.normalized_embedding[attr_id]
                logits = (torch.matmul(query,concept_embedding)/self.tau)
                log_prob = logits
            elif mutual_exclusive:
                prob, word2ix = self.query_attribute(query,attributes[attribute_index])


                concept_index = word2ix[identifier]

                log_prob = prob[:,concept_index]
            else:
                attr_identifier = attributes[attribute_index]
                mapping = self.get_attribute(attr_identifier)
                attr_id = self.attribute2id[attr_identifier]
            
                query = mapping(query)
                query = query / query.norm(2, dim=-1, keepdim=True)
                concept_embedding = concept.normalized_embedding[attr_id]
                logits = (torch.matmul(query,concept_embedding)/self.tau)
                log_prob = nn.LogSigmoid()(logits)



        


        
        return log_prob

    def similarity2(self, q1, q2, identifier, _normalized=False,logit_semantics=False):
        """
        Args:
            _normalized (bool): backdoor for function `cross_similarity`.
        """

        global _query_assisted_same

        logits_and = lambda x, y: torch.min(x, y)
        logits_or = lambda x, y: torch.max(x, y)

        tau = 0.1

        if not _normalized:
            q1 = q1 / q1.norm(2, dim=-1, keepdim=True)
            q2 = q2 / q2.norm(2, dim=-1, keepdim=True)

        if not _query_assisted_same or not self.training:
            #this is the code that runs during training. 
            margin = self._margin_cross
            logits = ((q1 * q2).sum(dim=-1)) / tau
            if logit_semantics:
                log_probs = logits
            else:
                log_probs = nn.LogSigmoid()(logits)
            return log_probs
        
    def kl_divergence(self,p1,p2):
        raw_p1 = torch.exp(p1)
        kl = raw_p1 * (p1-p2)
        kl = kl.sum(dim=-1)

        return kl

    def cross_similarity(self, query, identifier,logit_semantics=False):
        #identifier is an attribute, e.g. "color"
        #query is a tensor of object representations. a single object representation is a single vector
        if True:
            probs,word2idx = self.query_attribute(query,identifier)
            probs1,probs2 = jactorch.meshgrid(probs,dim=-2)

            kl = self.kl_divergence(probs1,probs2)

            return -1*kl




        else:
            mapping = self.get_attribute(identifier)
            query = mapping(query)
            query = query / query.norm(2, dim=-1, keepdim=True)
            q1, q2 = jactorch.meshgrid(query, dim=-2)
            #q1, q2 are used as all pairs of objects

            return self.similarity2(q1, q2, identifier, _normalized=True,logit_semantics=logit_semantics)

    def map_attribute(self, query, identifier):
        mapping = self.get_attribute(identifier)
        return mapping(query)


    # def query_attribute(self, query, identifier):
    #     #query is num_objs x obj_rep_size
    #     #identifier is an attribute
    #     #returns a log probability distribution over concepts, for each object (which concept is the answer to the query, for each object)
    #     mapping, concepts, attr_id = self.get_concepts_by_attribute(identifier)
    #     query = mapping(query)
    #     query = query / query.norm(2, dim=-1, keepdim=True)

    #     num_objs = query.size(0)

    #     word2idx = {}
    #     masks = []
    #     for k, v in concepts.items(): 
    #         belong_score = v.log_normalized_belong[attr_id]
    #         if belong_score < -50: #this concept does not belong to this attribute
    #             mask = -100.0*torch.ones((num_objs,),dtype=torch.float32).to(query.device)

    #         else:

    #             embedding = v.normalized_embedding[attr_id]
    #             embedding = jactorch.add_dim_as_except(embedding, query, -1)

    #             mask = ((query * embedding).sum(dim=-1) + self.margin)/self.tau
    #             #mask is 1xnum_objs


            
    #             #mask = mask + belong_score

    #         #mask is a num_objects list of log probabilities: the log probability that each object satisfies the concept

    #         masks.append(mask)
    #         word2idx[k] = len(word2idx)

    #     masks = torch.stack(masks, dim=-1)
    #     normalizing_constant = torch.logsumexp(masks,dim=1,keepdim=True)
    #     #we normalize the probabilities for each object: each object can only satisfy a single concept
        
    #     masks = masks - normalizing_constant

        

    #    return masks, word2idx
    def query_attribute(self, query, identifier):
        #query is num_objs x obj_rep_size
        #identifier is an attribute
        #returns a log probability distribution over concepts, for each object (which concept is the answer to the query, for each object)
        mapping, concepts, attr_id = self.get_concepts_by_attribute(identifier)
        query = mapping(query)
        #if not self.threshold_normalize:
        query = query / query.norm(2, dim=-1, keepdim=True)
        #else:
        #    query_norm = query.norm(2, dim=-1, keepdim=True)
        #    query = torch.where(query_norm<self.threshold_normalize, query, query / query_norm)

        num_objs = query.size(0)

        word2idx = {}
        masks = []
        embeddings = []
        skip_embedding = torch.zeros((query.size(1)),dtype=torch.float32).to(query.device)

        filter_keep = torch.zeros((query.size(0)),dtype=torch.float32).to(query.device)
        filter_remove = -100*torch.ones((query.size(0)),dtype=torch.float32).to(query.device)

        filters = []
        for k, v in concepts.items(): 
            belong_score = v.log_normalized_belong[attr_id]
            if belong_score < -50: #this concept does not belong to this attribute
                embedding = skip_embedding
                filters.append(filter_remove)

            else:

                embedding = v.normalized_embedding[attr_id]
                filters.append(filter_keep)
                #embedding = jactorch.add_dim_as_except(embedding, query, -1)

                #mask = ((query * embedding).sum(dim=-1) + self.margin)/self.tau
                #mask is 1xnum_objs


            
                #mask = mask + belong_score

            #mask is a num_objects list of log probabilities: the log probability that each object satisfies the concept

            embeddings.append(embedding)
            word2idx[k] = len(word2idx)

        filters = torch.stack(filters,dim=1)
        embeddings = torch.stack(embeddings, dim=1) 

        masks = (torch.matmul(query,embeddings)+ self.margin)/self.tau
        masks = masks+filters
        normalizing_constant = torch.logsumexp(masks,dim=1,keepdim=True)
        #we normalize the probabilities for each object: each object can only satisfy a single concept
        
        masks = masks - normalizing_constant

        

        return masks, word2idx