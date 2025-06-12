# SaLaMa
Local LLM assisted simultaneous writing

This is a quick and dirty vibecoded PoC

Allows editing and modifying text while LLM is producing more. If LLM digresses or goes off rails, stop it and restart it after removing / editing the part you didn't like.

Allows changing models 'on the fly' with dropdown selector. If one model is not able to continue the text properly, try another. 

For fiction writing, small (7-30Gb) models seem to be good for trying out ideas. Larger (70Gb+) models seem much better with nuances and more coherent plot and not-so-stereotypical character interactions.


Currently needs to be run in an environment with local [llama.cpp](https://github.com/ggml-org/llama.cpp) installation (assumes it's in the same directory).

Might be improved some day, or not. We'll see.
