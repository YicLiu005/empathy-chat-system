from pathlib import Path
import re
import math
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class RetrievedChunk:
    """
    A retrieved knowledge chunk from the local RAG knowledge base.
    """

    chunk_id: str
    source_file: str
    text: str
    score: float


class SimpleMarkdownRetriever:
    """
    A lightweight local retriever for markdown-based RAG knowledge base.

    It reads markdown files from knowledge_base/, splits them into chunks,
    and retrieves the most relevant chunks based on keyword overlap and
    simple TF-IDF-style scoring.

    This retriever does not call any LLM API.
    """

    def __init__(
        self,
        knowledge_base_dir: str = "knowledge_base",
        chunk_size: int = 350,
        chunk_overlap: int = 60,
    ):
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.chunks: List[Dict] = []
        self.idf: Dict[str, float] = {}

        self._load_knowledge_base()
        self._build_idf()

    def _load_knowledge_base(self) -> None:
        """
        Load all .md files from the knowledge base directory.
        """

        if not self.knowledge_base_dir.exists():
            raise FileNotFoundError(
                f"Knowledge base directory not found: {self.knowledge_base_dir}"
            )

        md_files = sorted(self.knowledge_base_dir.glob("*.md"))

        if not md_files:
            raise FileNotFoundError(
                f"No markdown files found in: {self.knowledge_base_dir}"
            )

        all_chunks = []

        for md_file in md_files:
            text = md_file.read_text(encoding="utf-8", errors="ignore")
            file_chunks = self._split_markdown_into_chunks(
                text=text,
                source_file=md_file.name,
            )
            all_chunks.extend(file_chunks)

        self.chunks = all_chunks

        if not self.chunks:
            raise ValueError("No valid chunks were created from the knowledge base.")

    def _split_markdown_into_chunks(
        self,
        text: str,
        source_file: str,
    ) -> List[Dict]:
        """
        Split markdown text into chunks.

        The function first tries to split by markdown headings.
        If a section is too long, it further splits by word length.
        """

        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

        if not text:
            return []

        sections = self._split_by_markdown_headings(text)

        chunks = []
        chunk_counter = 0

        for section in sections:
            section = section.strip()

            if not section:
                continue

            small_chunks = self._split_long_text(section)

            for small_chunk in small_chunks:
                small_chunk = small_chunk.strip()

                if not small_chunk:
                    continue

                chunk_counter += 1
                chunk_id = f"{source_file}::chunk_{chunk_counter}"

                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "source_file": source_file,
                        "text": small_chunk,
                        "tokens": self._tokenize(small_chunk),
                    }
                )

        return chunks

    def _split_by_markdown_headings(self, text: str) -> List[str]:
        """
        Split markdown document into sections by headings.
        """

        lines = text.split("\n")
        sections = []
        current_section = []

        for line in lines:
            if line.startswith("#") and current_section:
                sections.append("\n".join(current_section))
                current_section = [line]
            else:
                current_section.append(line)

        if current_section:
            sections.append("\n".join(current_section))

        return sections

    def _split_long_text(self, text: str) -> List[str]:
        """
        Split a long text into smaller word-based chunks.
        """

        words = text.split()

        if len(words) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunks.append(" ".join(chunk_words))

            if end >= len(words):
                break

            start = max(0, end - self.chunk_overlap)

        return chunks

    def _tokenize(self, text: str) -> List[str]:
        """
        Convert text into normalized tokens.
        """

        text = text.lower()

        tokens = re.findall(r"[a-zA-Z][a-zA-Z']*", text)

        stopwords = {
            "the", "a", "an", "and", "or", "but", "if", "then",
            "is", "are", "was", "were", "be", "been", "being",
            "to", "of", "in", "on", "for", "with", "as", "by",
            "at", "from", "this", "that", "these", "those",
            "it", "its", "i", "you", "he", "she", "they", "we",
            "my", "your", "his", "her", "their", "our",
            "do", "does", "did", "so", "not", "no",
        }

        return [token for token in tokens if token not in stopwords]

    def _build_idf(self) -> None:
        """
        Build IDF values for tokens across all chunks.
        """

        document_count = len(self.chunks)
        document_frequency: Dict[str, int] = {}

        for chunk in self.chunks:
            unique_tokens = set(chunk["tokens"])

            for token in unique_tokens:
                document_frequency[token] = document_frequency.get(token, 0) + 1

        self.idf = {}

        for token, df in document_frequency.items():
            self.idf[token] = math.log((document_count + 1) / (df + 1)) + 1

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        source_filter: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieve top-k relevant chunks for a user query.

        Args:
            query: User input text.
            top_k: Number of chunks to return.
            source_filter: Optional list of file names, for example:
                           ["sentiment_lexicon.md"] or ["sarcasm_examples.md"]

        Returns:
            A list of RetrievedChunk objects.
        """

        query = query.strip()

        if not query:
            return []

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        query_token_set = set(query_tokens)

        scored_results = []

        for chunk in self.chunks:
            if source_filter and chunk["source_file"] not in source_filter:
                continue

            score = self._score_chunk(
                query_tokens=query_tokens,
                query_token_set=query_token_set,
                chunk_tokens=chunk["tokens"],
                chunk_text=chunk["text"],
                source_file=chunk["source_file"],
            )

            if score > 0:
                scored_results.append(
                    RetrievedChunk(
                        chunk_id=chunk["chunk_id"],
                        source_file=chunk["source_file"],
                        text=chunk["text"],
                        score=score,
                    )
                )

        scored_results.sort(key=lambda x: x.score, reverse=True)

        return scored_results[:top_k]

    def _score_chunk(
        self,
        query_tokens: List[str],
        query_token_set: set,
        chunk_tokens: List[str],
        chunk_text: str,
        source_file: str,
    ) -> float:
        """
        Score a chunk based on token overlap and IDF weights.
        """

        if not chunk_tokens:
            return 0.0

        chunk_token_set = set(chunk_tokens)
        overlap_tokens = query_token_set.intersection(chunk_token_set)

        if not overlap_tokens:
            return 0.0

        score = 0.0

        for token in overlap_tokens:
            score += self.idf.get(token, 1.0)

        overlap_ratio = len(overlap_tokens) / max(len(query_token_set), 1)
        score += overlap_ratio * 2.0

        lower_query_words = set(query_tokens)
        lower_chunk_text = chunk_text.lower()

        for token in lower_query_words:
            if token in lower_chunk_text:
                score += 0.2

        if source_file == "sarcasm_examples.md":
            sarcasm_cues = {
                "great", "amazing", "perfect", "wonderful", "love",
                "thanks", "nice", "awesome", "brilliant", "fantastic",
                "obviously", "totally", "sure"
            }

            if query_token_set.intersection(sarcasm_cues):
                score += 1.5

        if source_file == "sentiment_lexicon.md":
            sentiment_cues = {
                "good", "bad", "great", "terrible", "excellent",
                "awful", "happy", "sad", "angry", "love", "hate",
                "positive", "negative", "disappointed", "satisfied"
            }

            if query_token_set.intersection(sentiment_cues):
                score += 1.0

        return score

    def get_stats(self) -> Dict:
        """
        Return basic statistics about the loaded knowledge base.
        """

        file_counts: Dict[str, int] = {}

        for chunk in self.chunks:
            source_file = chunk["source_file"]
            file_counts[source_file] = file_counts.get(source_file, 0) + 1

        return {
            "knowledge_base_dir": str(self.knowledge_base_dir),
            "total_chunks": len(self.chunks),
            "files": file_counts,
        }


if __name__ == "__main__":
    retriever = SimpleMarkdownRetriever(knowledge_base_dir="knowledge_base")

    print("Knowledge base loaded.")
    print(retriever.get_stats())

    while True:
        query = input("\nEnter query, or type 'exit': ").strip()

        if query.lower() == "exit":
            break

        results = retriever.retrieve(query, top_k=5)

        print("\nRetrieved Results:")
        for i, result in enumerate(results, start=1):
            print("=" * 60)
            print(f"Rank: {i}")
            print(f"Chunk ID: {result.chunk_id}")
            print(f"Source: {result.source_file}")
            print(f"Score: {result.score:.4f}")
            print(result.text[:800])