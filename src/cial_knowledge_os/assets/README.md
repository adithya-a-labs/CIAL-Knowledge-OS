# Packaged tiktoken vocabulary

`9b5ad71b2ce5302211f9c61530b329a4922fc6a4` is the official
`cl100k_base.tiktoken` vocabulary cached under the filename expected by
`tiktoken`.

- Source: `https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken`
- SHA-256: `223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7`
- Purpose: keep token counting and budgeting fully offline and deterministic.

The tiktoken constructor validates the expected SHA-256 before loading the
cached vocabulary.
