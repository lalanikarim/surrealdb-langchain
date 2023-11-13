import asyncio
from langchain.docstore.document import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from surrealdb import Surreal

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Collection,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
)


class SurrealDBStore(VectorStore):
    def __init__(self, dburl: str,
                 embeddings_function: Optional[Embeddings] = HuggingFaceEmbeddings(),
                 ns: str = "langchain",
                 db: str = "database",
                 collection: str = "documents",
                 **kwargs: Any) -> None:
        self.collection = collection
        self.ns = ns
        self.db = db
        self.dburl = dburl
        self.embeddings_function = embeddings_function
        self.score_threshold = kwargs.get("score_threshold", 0.7)
        self.sdb = Surreal()
        self.kwargs = kwargs

    async def initialize(self):
        await self.sdb.connect(self.dburl)
        if "db_user" in self.kwargs and "db_pass" in self.kwargs:
            user = self.kwargs.get("db_user")
            password = self.kwargs.get("db_pass")
            await self.sdb.signin({"user": user, "pass": password})

        await self.sdb.use(self.ns, self.db)

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return (
            self.embedding_function
            if isinstance(self.embedding_function, Embeddings)
            else None
        )

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        embeddings = self.embeddings_function.embed_documents(texts)
        ids = []
        for idx, text in enumerate(texts):
            record = await self.sdb.create(
                self.collection,
                {
                    "text": text,
                    "embedding": embeddings[idx]
                }
            )
            ids.append(record[0]["id"])
        return ids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        return asyncio.run(self.aadd_texts(texts, metadatas, **kwargs))

    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        query_embedding = self.embeddings_function.embed_query(query)
        results = await self.sdb.query(
            f"select id, text, vector::similarity::cosine(embedding,$embedding) as similarity from {self.collection} order by similarity desc LIMIT $k",
            {
                "embedding": query_embedding,
                "k": k
            }
        )
        return [
            Document(
                page_content=result["text"],
                metadata={"id": result["id"]}
            ) for result in results[0]["result"]
            if result["similarity"] >= self.score_threshold
        ]

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        async def _similarity_search():
            await self.initialize()
            return await self.asimilarity_search(query, k, **kwargs)
        return asyncio.run(_similarity_search())

    @classmethod
    async def afrom_texts(
        cls,
        dburl: str,
        texts: List[str],
        embeddings_function: Optional[Embeddings] = HuggingFaceEmbeddings(),
        ns: str = "langchain",
        db: str = "database",
        collection: str = "documents",
        **kwargs: Any,
    ) -> 'SurrealDBStore':
        sdb = cls(dburl, embeddings_function, ns, db, collection, **kwargs)
        await sdb.initialize()
        await sdb.aadd_texts(texts)
        return sdb

    @classmethod
    def from_texts(
        cls,
        dburl: str,
        texts: List[str],
        embeddings_function: Optional[Embeddings] = HuggingFaceEmbeddings(),
        ns: str = "langchain",
        db: str = "database",
        collection: str = "documents",
        **kwargs: Any,
    ) -> 'SurrealDBStore':
        sdb = asyncio.run(cls.afrom_texts(dburl, texts, embeddings_function,
                                          ns, db, collection, **kwargs))
        return sdb
