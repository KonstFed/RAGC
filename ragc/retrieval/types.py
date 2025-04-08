from ragc.retrieval.classical import PCSTConfig, SimpleEmbRetrievalConfig
from ragc.retrieval.common import NoRetrievalConfig

RetrievalConfig = SimpleEmbRetrievalConfig | PCSTConfig | NoRetrievalConfig
