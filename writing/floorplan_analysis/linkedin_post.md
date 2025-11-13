# LinkedIn Post: FloorPlan AI Benchmark

---

**Can AI read architectural drawings?**

We tested 6 frontier vision models (GPT-5, Claude, Gemini) on 15,000+ CAD floorplans to find out.

The task: Look at a technical drawing and identify which objects are present—doors, windows, fixtures, appliances.

**The results:**

✅ GPT-5-mini led at 58% detection (and only $0.07 per 100 images)
✅ All models consistently spotted doors (93%) and windows (80%)
❌ Small objects and text-labeled elements? Mostly missed (2-8% detection)
❌ 20-40% performance gap vs. specialized trained models

**What this means:**

Off-the-shelf vision AI can handle large, high-contrast architectural elements without training. But for production accuracy (80-90%+), you still need domain-specific models or human-in-the-loop workflows.

The zero-shot gap is real. Technical drawings aren't photographs—they're abstract, conventional symbols. General-purpose models haven't learned architectural language yet.

**Key takeaway:** If you're building AI for construction, set realistic expectations. Test on your own drawings. Design for incomplete detection. And consider fine-tuning when accuracy matters.

We ran this experiment to understand what works today vs. what requires more specialized approaches.

Read the full benchmark (with detailed category breakdowns and practical recommendations): [LINK]

---

*At AIDE, we're building AI agents for the built world. Interested in exploring how AI can work for your construction business? Let's talk: https://go.getaide.ai/demo*

#AI #Construction #ComputerVision #MachineLearning #ConstructionTech
