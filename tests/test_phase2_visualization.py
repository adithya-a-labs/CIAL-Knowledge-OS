from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from cial_knowledge_os.visualization import (
    batch_retrieval_trace_table,
    citation_quality_table,
    context_stage_counts_table,
    display_context_sections_table,
    display_high_duplicate_chunks_table,
    display_low_score_chunks_table,
    display_query_variant_contribution_table,
    display_retrieval_trace_table,
    display_top_sources_table,
    duplicate_chunk_frequency_table,
    neighbor_expansion_table,
    plot_answer_status_distribution,
    plot_context_compression_ratio,
    plot_context_section_lengths,
    plot_context_stage_counts,
    plot_duplicate_chunk_frequency,
    plot_latency_by_question,
    plot_page_distribution,
    plot_query_variant_contribution,
    plot_retrieval_comparison,
    plot_retrieval_funnel,
    plot_retrieval_scores,
    plot_score_by_query_variant,
    plot_score_distribution,
    plot_source_distribution,
    query_variants_table,
    retrieval_chunks_table,
    retrieval_comparison_table,
)


def _result(
    chunk_index: int,
    *,
    score: float = 0.7,
    is_neighbor: bool = False,
    source: str = "manual.pdf",
    page: int = 7,
    matched_queries: tuple[str, ...] = ("original",),
) -> dict[str, object]:
    chunk_id = f"{source}:p{page}:c{chunk_index}"
    return {
        "text": f"Evidence for chunk {chunk_index}.",
        "score": score,
        "source": source,
        "page_number": page,
        "chunk_id": chunk_id,
        "is_neighbor": is_neighbor,
        "metadata": {
            "source": f"C:/corpus/{source}",
            "file_name": source,
            "page_number": page,
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
        },
        "matched_queries": list(matched_queries),
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
                "retrieval_latency_seconds": 0.4,
                "answer_latency_seconds": 0.8,
                "total_latency_seconds": 1.3,
                "retrieval_trace": "Original Query → Retrieved 34 chunks",
            }
        ]

        citation_table = citation_quality_table(citations)
        batch_table = batch_retrieval_trace_table(rows)

        self.assertEqual(citation_table.iloc[0]["document"], "manual.pdf")
        self.assertEqual(citation_table.iloc[0]["similarity_score"], 0.75)
        self.assertEqual(batch_table.iloc[0]["chunks_before_deduplication"], 34)
        self.assertEqual(batch_table.iloc[0]["total_latency_seconds"], 1.3)
        self.assertIn("Retrieved 34", batch_table.iloc[0]["retrieval_trace"])

    def test_additional_debugging_tables_use_phase2_trace_data(self) -> None:
        original = _result(1, score=0.8)
        duplicate = _result(1, score=0.75, matched_queries=("rewritten",))
        weak = _result(
            2,
            score=0.3,
            source="operations.pdf",
            page=12,
            matched_queries=("keyword_expanded",),
        )
        final = _result(
            1,
            score=0.8,
            matched_queries=("original", "rewritten"),
        )
        final["text"] = "Final section text."
        trace = {
            "context_stages": {
                "retrieved": [original, duplicate, weak],
                "deduplicated": [original, weak],
                "expanded": [original, weak],
                "merged": [original, weak],
                "compressed": [final],
            },
            "context": "Final formatted context.",
        }

        contribution = display_query_variant_contribution_table(trace)
        sources = display_top_sources_table(trace["context_stages"]["retrieved"])
        sections = display_context_sections_table(trace)
        low_scores = display_low_score_chunks_table(
            trace["context_stages"]["retrieved"],
            threshold=0.5,
        )
        duplicates = display_high_duplicate_chunks_table(
            trace["context_stages"]["retrieved"]
        )

        self.assertEqual(
            set(contribution["query_variant"]),
            {"original", "rewritten", "keyword_expanded"},
        )
        self.assertEqual(sources.iloc[0]["document"], "manual.pdf")
        self.assertEqual(sections.iloc[0]["section_characters"], 19)
        self.assertEqual(low_scores.iloc[0]["chunk_id"], weak["chunk_id"])
        self.assertEqual(duplicates.iloc[0]["frequency"], 2)

    def test_additional_phase2_plots_render_from_synthetic_traces(self) -> None:
        original = _result(1, score=0.8)
        rewritten = _result(
            2,
            score=0.65,
            source="operations.pdf",
            page=12,
            matched_queries=("rewritten",),
        )
        final = dict(original)
        final["text"] = "Short final evidence."
        final["matched_queries"] = ["original", "rewritten"]
        trace = {
            "context_stages": {
                "retrieved": [original, rewritten],
                "deduplicated": [original, rewritten],
                "expanded": [original, rewritten],
                "merged": [original, rewritten],
                "compressed": [final],
            },
            "context": "Final prompt context.",
        }
        batch_rows = [
            {
                "question": "Fast question?",
                "answer_status": "Answered",
                "total_latency_seconds": 0.5,
                "retrieval_trace": "Retrieved 2 chunks",
            },
            {
                "question": "Slow unsupported question?",
                "answer_status": "Insufficient Evidence",
                "total_latency_seconds": 2.5,
                "retrieval_trace": "Retrieved 0 chunks",
            },
        ]

        axes = [
            plot_query_variant_contribution(trace),
            plot_source_distribution(trace["context_stages"]["retrieved"]),
            plot_page_distribution(trace["context_stages"]["retrieved"]),
            plot_score_distribution(trace["context_stages"]["retrieved"]),
            plot_score_by_query_variant(trace),
            plot_context_compression_ratio(trace),
            plot_context_section_lengths(trace),
            plot_retrieval_funnel(trace),
            plot_answer_status_distribution(batch_rows),
            plot_latency_by_question(batch_rows),
        ]
        trace_table = display_retrieval_trace_table(batch_rows)

        self.assertTrue(all(axis.get_title() for axis in axes))
        self.assertEqual(len(trace_table), 2)
        self.assertIn("total_latency_seconds", trace_table.columns)

    def test_visualization_helpers_handle_empty_inputs(self) -> None:
        empty_trace = {"context_stages": {}, "context": ""}
        tables = [
            query_variants_table([]),
            retrieval_chunks_table([], stage="empty"),
            retrieval_comparison_table([], []),
            duplicate_chunk_frequency_table([]),
            neighbor_expansion_table([], []),
            context_stage_counts_table(empty_trace),
            citation_quality_table([]),
            batch_retrieval_trace_table([]),
            display_query_variant_contribution_table(empty_trace),
            display_top_sources_table([]),
            display_context_sections_table(empty_trace),
            display_retrieval_trace_table([]),
            display_low_score_chunks_table([]),
            display_high_duplicate_chunks_table([]),
        ]
        plots = [
            plot_retrieval_scores([]),
            plot_retrieval_comparison([], []),
            plot_duplicate_chunk_frequency([]),
            plot_context_stage_counts(empty_trace),
            plot_query_variant_contribution(empty_trace),
            plot_source_distribution([]),
            plot_page_distribution([]),
            plot_score_distribution([]),
            plot_score_by_query_variant(empty_trace),
            plot_context_compression_ratio(empty_trace),
            plot_context_section_lengths(empty_trace),
            plot_retrieval_funnel(empty_trace),
            plot_answer_status_distribution([]),
            plot_latency_by_question([]),
        ]

        self.assertTrue(all(isinstance(table, pd.DataFrame) for table in tables))
        self.assertTrue(all(plot is not None for plot in plots))


if __name__ == "__main__":
    unittest.main()
