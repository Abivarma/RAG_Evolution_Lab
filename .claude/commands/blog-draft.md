Generate a first draft of the blog post for stage N.

Usage: /blog-draft <N>

Steps:
1. Load the most recent eval JSON from results/stage_<N>/
2. Load the blog outline from SPEC.md section 6.<N+1>
3. Generate a draft with:
   - Hook (use exact LinkedIn hook from spec, substituting real numbers)
   - Problem section
   - Architecture diagram (ASCII from SPEC.md)
   - Key numbers table (real benchmark results)
   - Trade-off analysis vs previous stage
   - Where this architecture wins / loses
   - Teaser for next post
4. Save to docs/blog/post_<N+1>_draft.md
5. Print word count and flag any [NUMBER] placeholders still needing real values.
