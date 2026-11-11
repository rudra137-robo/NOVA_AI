import unittest

from app.knowledge.chunker import (
    clean_text,
    chunk_text
)


class TestChunker(unittest.TestCase):

    def test_empty_text(self):
        self.assertEqual(
            chunk_text(""),
            []
        )

    def test_single_chunk(self):

        result = chunk_text(
            "Hello SASTRA AI",
            chunk_size=100,
            overlap=10
        )

        self.assertEqual(
            result,
            ["Hello SASTRA AI"]
        )

    def test_chunk_overlap(self):

        text = "A" * 1000

        result = chunk_text(
            text,
            chunk_size=400,
            overlap=100
        )

        self.assertEqual(len(result), 3)

        self.assertEqual(
            len(result[0]),
            400
        )

        self.assertEqual(
            result[0][-100:],
            result[1][:100]
        )

    def test_invalid_overlap(self):

        with self.assertRaises(ValueError):
            chunk_text(
                "Test",
                chunk_size=100,
                overlap=100
            )


if __name__ == "__main__":
    unittest.main()