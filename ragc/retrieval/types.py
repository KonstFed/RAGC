from ragc.retrieval.classical import PCSTConfig, SimpleEmbRetrievalConfig
from ragc.retrieval.common import NoRetrievalConfig
from ragc.retrieval.gnn import GNNRetrievalConfig

RetrievalConfig = SimpleEmbRetrievalConfig | PCSTConfig | NoRetrievalConfig | GNNRetrievalConfig
