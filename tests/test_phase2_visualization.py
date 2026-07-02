from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from cial_knowledge_os.visualization import (
    batch_retrieval_trace_table,
    citation_quality_table,
    context_stage_counts_table,
    duplicate_chunk_frequency_table,
    neighbor_expansion_table,
    plot_context_stage_counts,
    plot_duplicate_chunk_frequency,
    plot_retrieval_comparison,
    query_variants_table,
    retrieval_chunks_table,
    retrieval_comparison_table,
)


def _result(
    chunk_index: int,
    *,
    score: float = 0.7,
    is_neighbor: bool = False,
) -> dict[str, object]:
    chunk_id = f"manual:p7:c{chunk_index}"
    return {
        "text": f"Evidence for chunk {chunk_index}.",
        "score": score,
        "source": "manual.pdf",
        "page_number": 7,
        "chunk_id": chunk_id,
        "is_neighbor": is_neighbor,
        "metadata": {
            "source": "C:/corpus/manual.pdf",
            "file_name": "manual.pdf",
            "page_number": 7,
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
        },
        "matched_queries": ["original"],
    }


class Phase2VisualizationTests(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_query_and_retrieval_tables_use_trace_values(self) -> None:
        variants = [
            {"technique": "original", "query": "Original question"},
            {"technique": "rewritten", "query": "Rewritten question"},
        ]
        variant_table = query_variants_table(variants)
        chunk_table = retrieval_chunks_table(
            [_result(1)],
            stage="deduplicated",
        )

        self.assertEqual(variant_table["query"].tolist(), [
            "Original question",
            "Rewritten question",
        ])
        self.assertFalse(bool(variant_table.iloc[0]["changed_from_original"]))
        self.assertTrue(bool(variant_table.iloc[1]["changed_from_original"]))
        self.assertEqual(chunk_table.iloc[0]["document"], "manual.pdf")
        self.assertEqual(chunk_table.iloc[0]["stage"], "deduplicated")

    def test_comparison_and_duplicate_frequency_reflect_input(self) -> None:
        duplicate = _result(1)
        raw = [duplicate, duplicate, _result(2)]
        deduplicated = [_result(1), _result(2)]

        comparison = retrieval_comparison_table(raw, deduplicated)
        frequency = duplicate_chunk_frequency_table(raw)

        self.assertEqual(comparison.iloc[0]["returned_chunks"], 3)
        self.assertEqual(comparison.iloc[0]["unique_chunks"], 2)
        self.assertEqual(frequency.iloc[0]["frequency"], 2)
        self.assertTrue(bool(frequency.iloc[0]["is_duplicate"]))

        comparison_axis = plot_retrieval_comparison(raw, deduplicated)
        frequency_axis = plot_duplicate_chunk_frequency(raw)
        self.assertIn("multi-query", comparison_axis.get_title())
        self.assertIn("Duplicate chunk", frequency_axis.get_title())

    def test_neighbor_and_context_tables_show_stage_changes(self) -> None:
        seed = _result(2)
        neighbor = _result(1, is_neighbor=True)
        neighbor["neighbor_offset"] = -1
        neighbor["seed_chunk_id"] = seed["chunk_id"]
        trace = {
            "context_stages": {
                "retrieved": [seed, seed],
                "deduplicated": [seed],
                "expanded": [neighbor, seed],
                "merged": [seed],
                "compressed": [seed],
            },
            "context": "Final context",
        }

        neighbor_table = neighbor_expansion_table(
            trace["context_stages"]["deduplicated"],
            trace["context_stages"]["expanded"],
        )
        context_table = context_stage_counts_table(trace)
        axis = plot_context_stage_counts(trace)

        self.assertEqual(
            set(neighbor_table["expansion_role"]),
            {"retrieved_seed", "added_adjacent_chunk"},
        )
        self.assertEqual(context_table["section_count"].tolist(), [2, 1, 2, 1, 1])
        self.assertEqual(
            int(context_table.iloc[-1]["final_context_characters"]),
            len("Final context"),
        )
        self.assertIn("final context", axis.get_title().casefold())

    def test_citation_and_batch_trace_tables_preserve_audit_fields(self) -> None:
        citations = [
            {
                "reference_id": 1,
                "source_file": "manual.pdf",
                "source_path": "C:/corpus/manual.pdf",
                "page_number": 7,
                "chunk_id": "manual:p7:c1",
                "score": 0.75,
            }
        ]
        rows = [
            {
                "question": "Question?",
                "answer_status": "Answered",
                "chunks_before_deduplication": 34,
                "chunks_after_deduplication": 19,
                "chunks_after_neighbor_expansion": 28,
                "final_context_sections": 7,
                "final_context_characters": 2000,
                "retrieval_trace": "Original Query → Retrieved 34 chunks",
            }
        ]

        citation_table = citation_quality_table(citations)
        batch_table = batch_retrieval_trace_table(rows)

        self.assertEqual(citation_table.iloc[0]["document"], "manual.pdf")
        self.assertEqual(citation_table.iloc[0]["similarity_score"], 0.75)
        self.assertEqual(batch_table.iloc[0]["chunks_before_deduplication"], 34)
        self.assertIn("Retrieved 34", batch_table.iloc[0]["retrieval_trace"])


if __name__ == "__main__":
    unittest.main()
