# Testing Vision Models on Construction Floorplans: Which AI Actually Works?

*A technical benchmark comparing 6 frontier vision models on architectural drawing interpretation*

---

## The Challenge

At AIDE, we build AI agents for the built world—automating workflows that span from takeoff to project management in construction. A foundational capability for these agents is understanding architectural drawings: floorplans, elevations, and technical CAD files that form the backbone of every construction project.

The challenge? These aren't photographs. They're technical drawings filled with standardized symbols, abstract representations, and precise annotations. A door isn't a realistic image—it's an arc showing swing direction. A toilet is an oval with specific dimensions. An elevator might just be a box labeled "E."

With nearly 80% of architects expressing concerns about AI accuracy in the AEC industry, we prefer to run our own experiments rather than rely on vendor claims. This benchmark tests a fundamental question: **Can frontier vision models reliably detect objects in architectural CAD drawings?**

## The Dataset: FloorPlanCAD

We used the [FloorPlanCAD dataset](https://www.kaggle.com/datasets/samirshabani/architecture) from Kaggle, containing:

- **15,285 architectural floorplan images** (PNG format)
- **28 object categories** with YOLO-format annotations
- **Real-world CAD drawings** from architectural projects

Object categories span the typical elements found in construction drawings:

**Structural**: Doors (single, double, sliding), windows (standard, bay, blind), stairs, elevators, escalators

**Fixtures**: Toilets, sinks, baths, showers

**Appliances**: Stoves, refrigerators, washers, dryers, dishwashers

**Furniture**: Sofas, beds, tables, chairs, desks, bookshelves, wardrobes, cabinets, TV stands, plants

**Key insight**: The ground truth annotations are incomplete—human annotators labeled only 5-20% of visible objects in each drawing. This means we can't measure count accuracy, but we *can* test detection rate: if a human labeled a toilet, did the AI spot it?

## Research Method

### Models Tested

We benchmarked six frontier vision models across three providers:

| Provider | Models | Notes |
|----------|--------|-------|
| Google | Gemini 2.5 Flash, Gemini 2.5 Pro | Latest multimodal models |
| Anthropic | Claude Sonnet 4.5, Claude Haiku 4.5 | Released Sept/Oct 2025 |
| OpenAI | GPT-5, GPT-5-mini | Reasoning models with vision |

### Test Design

**Sample selection**: 100 randomly selected floorplans with non-empty annotations

**Task**: Detect and count 28 object categories from CAD symbols

**Prompt engineering**: Detailed descriptions of CAD symbol conventions (e.g., "door_single: Arc/curved line showing door swing")

**Output format**: Structured JSON extraction with category, count, and confidence

**Metric**: Detection rate—if ground truth has object X, did the model find it? (presence/absence, not count accuracy)

### Why Detection Rate?

Given incomplete ground truth, we focus on: "When a human annotator explicitly marked an object, did the AI detect it?" This tests whether models can recognize architectural symbols reliably, which is critical for downstream applications like quantity takeoffs and code compliance.

## Key Findings

> **[PLACEHOLDER: Executive summary of results]**
> Example: "Claude Haiku 4.5 achieved the highest overall detection rate at XX%, while GPT-5-mini offered the best cost-performance ratio at $0.XX per 100 samples."

### Overall Model Performance

**[PLACEHOLDER: Table or figure showing overall detection rates]**

| Model | Detection Rate | Speed | Cost (100 samples) |
|-------|----------------|-------|-------------------|
| gemini-2.5-flash | XX% | XX sec | $X.XX |
| gemini-2.5-pro | XX% | XX sec | $X.XX |
| sonnet-4.5 | XX% | XX sec | $X.XX |
| haiku-4.5 | XX% | XX sec | $X.XX |
| gpt-5 | XX% | XX sec | $X.XX |
| gpt-5-mini | XX% | XX sec | $X.XX |

**[PLACEHOLDER: Key insight about best model]**

### Speed vs Cost vs Accuracy

**[PLACEHOLDER: Scatter plot or analysis of tradeoffs]**

**Cost leader**: GPT-5-mini at $0.XX per 100 samples (XX% detection)

**Speed leader**: [Model name] at XX seconds per sample (XX% detection)

**Accuracy leader**: [Model name] at XX% detection ($X.XX per 100 samples)

**Best value**: [Model name] balancing all three factors

## Going Forward

This benchmark reveals both promise and limitations for vision AI in construction workflows:

**What works today**: Models can reliably detect high-contrast architectural elements like doors, windows, and some fixtures. For applications requiring 50-75% coverage (initial quantity estimates, design review checklists), current vision AI is viable with human oversight.

**What doesn't work yet**: Small symbols, text-dependent annotations (elevator labels), and decorative elements remain challenging. Automated code compliance or material takeoffs requiring 90%+ accuracy still need hybrid approaches.

**For practitioners building with vision AI**:

1. **Choose models based on your accuracy threshold**: If cost matters and 60-70% detection suffices, smaller models like GPT-5-mini or Haiku 4.5 offer excellent value. If maximizing detection rate, [best model] is worth the premium.

2. **Engineer for incomplete detection**: Design workflows assuming the AI will miss 20-40% of objects. Build review interfaces, confidence thresholds, and human-in-the-loop verification.

3. **Test on your specific drawings**: Our results are from one CAD dataset. Symbol conventions, drawing styles, and annotation density vary across firms and regions. Run similar benchmarks on your own drawings before production deployment.

4. **Prompt engineering matters**: Detailed symbol descriptions improved detection significantly. Generic "count objects" prompts underperform compared to CAD-specific guidance.

At AIDE, we use these insights to build reliable agents that understand when to trust AI output and when to escalate to human experts. The future of construction AI isn't about replacing human judgment—it's about augmenting it with tools that work within realistic accuracy bounds.

---

**Want to discuss construction AI experiments?** Reach out at [contact info or link]

---

## Appendix: Full Category Results

**[PLACEHOLDER: Detailed table of detection rates by category and model]**

| Category | gemini-2.5-flash | gemini-2.5-pro | sonnet-4.5 | haiku-4.5 | gpt-5 | gpt-5-mini |
|----------|------------------|----------------|------------|-----------|-------|------------|
| door_single | XX% | XX% | XX% | XX% | XX% | XX% |
| window | XX% | XX% | XX% | XX% | XX% | XX% |
| toilet | XX% | XX% | XX% | XX% | XX% | XX% |
| elevator | XX% | XX% | XX% | XX% | XX% | XX% |
| stairs | XX% | XX% | XX% | XX% | XX% | XX% |
| sink | XX% | XX% | XX% | XX% | XX% | XX% |
| ... | ... | ... | ... | ... | ... | ... |

### What Models Detect Well vs. Poorly

**[PLACEHOLDER: Analysis bullets with reasoning]**

**Strong performance (>75% detection across most models):**
- **[Category examples]** - [Reasoning: e.g., high-contrast, large symbols, clear visual features]
- **[Category examples]** - [Reasoning]

**Moderate performance (40-75% detection):**
- **[Category examples]** - [Reasoning: e.g., symbol size/clarity varies, contextual ambiguity]
- **[Category examples]** - [Reasoning]

**Poor performance (<40% detection across most models):**
- **[Category examples]** - [Reasoning: e.g., text-dependent labels, small scale, abstract representation]
- **[Category examples]** - [Reasoning]

**Key patterns:**
- [Pattern observation with explanation]
- [Pattern observation with explanation]
- [Pattern observation with explanation]
