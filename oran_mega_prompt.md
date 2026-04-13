### SECTION 1: GLOBAL CONTEXT & COGNITIVE ARCHITECTURE

**To the AI Assistant:**
You are acting as an expert Machine Learning Engineer and Telecommunications Researcher. You are writing a complete, production-ready Python simulation for an IEEE Communications Magazine paper. The project is an **O-RAN Inspired Cognitive Architecture for 6G (RIS + NOMA)**.

**1.1 Project Paradigm (CRITICAL):**
Unlike standard Deep Reinforcement Learning (DRL) that memorizes numeric states, this system uses a multi-agent LLM orchestration loop to distill physics outputs into **Generalized Semantic Concepts**. The agent learns from both successes and failures (without arbitrary utility caps) and stores these heuristics in a Concept-RAG database.

**1.2 Strict Architectural Mandates:**
* **Modularity:** The codebase MUST be split into exactly 5 files: `memory_manager.py`, `physics_env.py`, `agent_orchestrator.py`, `visualizer.py`, and `main.py`.
* **Hardware & Model:** The LLM is **`deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`** running on an NVIDIA A100. Load the model using `BitsAndBytesConfig` in 4-bit or 8-bit to ensure high speed and VRAM safety.
* **Absolute Honesty:** NEVER hardcode performance metrics, fake iterations, or cap utility scores. The math solver must be the absolute ground truth.
* **Terminology:** Phase 0 is called "Baseline Evaluation," NOT "Cold Start."

**1.3 Hierarchical Scenario Pre-Generation (`main.py` setup):**
Before the continual learning loop starts, `main.py` MUST generate a strictly hierarchical dataset of scenarios.
* **The 3 Categories:** `RIS_ONLY`, `NOMA_ONLY`, and `JOINT`.
* **The Unseen Test Clusters:** For each category, define ONE completely unique physical cluster (e.g., "Extreme Diagonal Blockage"). Generate exactly 1 fixed scenario from this unseen cluster to serve as `TEST_RIS`, `TEST_NOMA`, and `TEST_JOINT`. The agent MUST NEVER train on these clusters.
* **The Training Clusters:** For each category, define 2 to 3 *different* physical clusters (e.g., `Cluster_Center_Blockage`, `Cluster_High_Correlation`). Inside each training cluster, generate a list of 3 to 5 similar scenarios (e.g., slight coordinate variations). This forces the Evaluator Agent to deduce stable concepts from variations, while ensuring the Test Scenarios remain a true Zero-Shot evaluation.

**The 5-Phase Execution Flow (Epoch-Based):**
* **Phase 0 (Baseline):** Evaluate the 3 Test Scenarios with an empty database. (Acts as Epoch 0).
* **Phase 1 (RIS Learning):** Train on RIS scenarios for a configurable number of epochs. Evaluate on `TEST_RIS` at the end of *each* epoch to track learning progression. 
* **Phase 2 (NOMA Learning):** Train on NOMA scenarios for a configurable number of epochs. Evaluate on `TEST_NOMA` at the end of *each* epoch.
* **Phase 3 (Zero-Shot Composition):** Evaluate `TEST_JOINT` using the merged database. NO training occurs here.
* **Phase 4 (Joint Mastery):** Train on Joint scenarios for a configurable number of epochs. Evaluate on `TEST_JOINT` at the end of *each* epoch.

### SECTION 2: INDEPENDENT CONCEPT-RAG DATABASES (`memory_manager.py`)

**To the AI Assistant:**
Write the complete code for `memory_manager.py`. This acts as the "Librarian's Brain." To simulate independent O-RAN domains, this module must support instantiating completely separate databases that can later be merged.

**2.1 Core Class Initialization (`ConceptDatabase`):**
* Initialize `self.memory = []` to hold concept dictionaries.
* Initialize an embedding model (e.g., `SentenceTransformer('all-MiniLM-L6-v2', device='cuda')`).
* Accept a `db_path` string on initialization (e.g., `snapshots/ris_db.pkl`). 
* **Auto-Load Logic:** If `db_path` exists, load the `.pkl` into `self.memory`.

**2.2 The Merge Function (CRITICAL FOR PHASE 3):**
* Implement a `merge_with(self, other_database)` method.
* This method must iterate through the `other_database.memory` and append its concepts to `self.memory`. 
* After merging, it must immediately save the newly fused database to its own `db_path`.

**2.3 Learning & Continuous Persistence (`learn_concept`):**
* **Inputs:** `condition` (the network state), `rule` (the heuristic), and `utility_score`.
* Create a composite string (Condition + Rule) and embed it.
* **The Update Logic:** Compute cosine similarity against existing concepts in this specific database instance.
    * If `Max Similarity > 0.85`: If the new `utility_score` is higher, OVERWRITE the old rule. If it's worse, discard it.
    * If `Max Similarity <= 0.85`: Append as a novel concept to `self.memory`.
* **Auto-Save:** At the end of this method, immediately call `self._save_to_disk()` so the newly learned rule is permanently stored in the assigned `db_path`.

**2.4 XAI Artifact Export (`export_to_markdown`):**
* **Export:** Write a method that reads `self.memory`. It must accept an `llm_client` or `llm_pipeline` parameter passed from `main.py` so it does not reload the model into VRAM. Pass the raw dictionary data to this Formatter Agent.... 
* **Formatter Agent Prompt:** Instruct the LLM to format the raw concepts into a beautiful O-RAN style Markdown file (with headers for Policy, Condition, and Rule). **CRITICAL INSTRUCTION:** The LLM must NEVER change or rewrite the scientific wording of the concepts; it must only add structural Markdown formatting. Save the output to the `artifacts/` folder.


### SECTION 3: THE PHYSICS SOLVER (`physics_env.py`)

**To the AI Assistant:**
Write the complete code for `physics_env.py`. This is the ground truth of the simulation. It calculates the raw physical layer metrics (Sum-rate, EEPSU) based strictly on the parameters proposed by the LLM. 

**3.1 The Math Solver:**
* **No Iteration Faking:** The mathematical solver acts as an instant calculator. Do not add artificial iterations or delays here. It simply takes the `params` (e.g., `noma_power_split`, `ris_phase_matrix`), runs the interference equations, and returns the `sum_rate`, `eepsu`, and QoS. Use standard 6G mathematical models: model the RIS using a cascaded Base Station-to-RIS-to-User channel with phase shifts, and model NOMA using Successive Interference Cancellation (SIC) where the near user decodes and subtracts the far user's signal before decoding its own.

**3.2 Domain-Specific Utility Scores ($U$):**
Implement three distinct, rigorous utility functions, as RIS and NOMA have completely different physical objectives in isolation:
* `calc_ris_utility()`: Evaluates RIS isolation. Formula: $U_{RIS} = \alpha \left(\frac{\text{SNR}_{\text{achieved}}}{\text{SNR}_{\text{target}}}\right) + \beta(\text{Reflection Efficiency})$. 
* `calc_noma_utility()`: Evaluates NOMA isolation. Formula: $U_{NOMA} = \alpha(\text{Jain's Fairness Index}) + \beta(\text{Sum-Rate}) - \gamma(\text{SIC Error Penalty})$.
* `calc_joint_utility()`: Evaluates the merged system. Formula: $U_{JOINT} = \alpha \left(\frac{QoS_{\text{achieved}}}{QoS_{\text{target}}}\right) + \beta(EEPSU)$.
**CRITICAL:** There is no artificial cap or `max()` function used to fake these scores. The physics engine must dynamically call the correct formula based on the current scenario type.




### SECTION 4: THE 3-AGENT LOOP (`agent_orchestrator.py`)

**To the AI Assistant:**
Write the code for `agent_orchestrator.py`. This contains the Cognitive Loop where the LLM actually "thinks" and learns. 

**4.1 Handling DeepSeek-R1 Reasoning Tags (CRITICAL):**
* DeepSeek-R1 models will output internal reasoning inside `<think> ... </think>` tags before outputting the final JSON. 
* **You MUST write a regex cleaning function** (e.g., `re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)`) to strip these tags out before attempting to parse the LLM's output using `json.loads()`.

**4.2 The Coordinator Agent (The Pilot):**
* **Input:** Scenario Description, Active Domains (e.g., `["ris", "noma"]`).
* **Action:** Queries `memory_manager.retrieve_concepts()` to get the relevant rules. Synthesizes these rules with the current scenario to generate a numeric parameter dictionary for the Physics Solver. It must return pure JSON.

**4.3 The Evaluator Agent (The Scientist):**
* **Input:** Scenario, Proposed Params, Resulting Utility Score ($U$), Sum-rate.
* **Action:** Regardless of whether the Utility is 0.1 (Failure) or 0.9 (Success), this agent must write a 1-2 sentence human-readable Explanation/Concept detailing *why* the physics behaved that way (e.g., "Bounding NOMA power below 0.5 prevented interference"). It must return pure JSON.

**4.4 The Librarian Agent (The Knowledge Manager):**
* **Input:** The final lesson/concept generated by the Evaluator Agent.
* **Action:** Analyzes the physics and classifies the concept's domain (tagging it as `"ris"`, `"noma"`, or `"joint"`). **Note:** The Orchestrator will be passed ONLY ONE active database instance per phase. The Librarian formats the concept and simply calls `learn_concept()` on that single active database.

**4.5 The Cognitive Training Loop (`run_agentic_optimization`):**
* For a given training scenario, run a `while` loop (Maximum 10 Agent Iterations).
* Step 1: Coordinator proposes parameters.
* Step 2: Physics solver calculates $U$.
* Step 3: If $U \ge 0.70$, **BREAK** the loop (Success).
* Step 4: If $U < 0.70$, feed the failure back to the Coordinator to try again.
* Step 5: After the loop breaks (or hits max 10), trigger the **Evaluator Agent** to write the final lesson. Pass that lesson to the **Librarian Agent**, which tags it and updates the single `active_db` for that phase via `learn_concept()`. Return the number of Agent Iterations it took.




### SECTION 5: THE CONTINUAL LEARNING MASTER (`main.py`)

**To the AI Assistant:**
Write the code for `main.py`. This orchestrates the absolute chronological progression of the phases, explicitly handling the independent creation and subsequent merging of the RAG databases.

**5.1 Global Logging & Setup:**
* Define a configurable parameter `NUM_EPOCHS = 5` at the top of the script.
* Log `epoch`, `scenario_index`, `phase_name`, `agent_iterations`, `sum_rate`, `noma_u1_power_ratio`, `ris_u3_snr`, `scenario_type` (e.g., `'train'`, `'test_ris'`), and `domain_utility_score` to a CSV.
 
**5.2 Database Instantiation & The 5-Phase Pipeline:**
Instantiate three distinct databases at the start of the script: `ris_db`, `noma_db`, `joint_db`.
* **Phase 0 (Baseline):** Evaluate `TEST_RIS`, `TEST_NOMA`, and `TEST_JOINT` using empty databases. Log as `epoch 0`.
* **Phase 1 (RIS Only Learning):** Wrap in `for epoch in range(1, NUM_EPOCHS + 1):`. Inside the loop, train on all `RIS_ONLY` clusters using `ris_db`. After the training loop for that epoch finishes, evaluate `TEST_RIS` to track progression.
* **Phase 2 (NOMA Only Learning):** Wrap in `for epoch in range(1, NUM_EPOCHS + 1):`. Train on all `NOMA_ONLY` clusters using `noma_db`. After each epoch, evaluate `TEST_NOMA`.
* **Phase 3 (Zero-Shot Database Merge):** Initialize `joint_db`. Load `ris_db` into it, then call `joint_db.merge_with(noma_db)`. Evaluate `TEST_JOINT` using `joint_db`. DO NOT train here. Log as a distinct "Zero-Shot" event.
* **Phase 4 (Joint Mastery):** Wrap in `for epoch in range(1, NUM_EPOCHS + 1):`. Train on all `JOINT` clusters using ONLY `joint_db`. Evaluate `TEST_JOINT` after each epoch.


**5.3 Final Artifact Generation:**
* At the very end of `main.py`, you MUST call `.export_to_markdown()` on all three databases to generate: `artifacts/ris_rulebook.md`, `artifacts/noma_rulebook.md`, and `artifacts/joint_rulebook.md`.
* Call the `visualizer.py` dashboard to generate the paper's plots.



### SECTION 6: PUBLICATION PLOTS (`visualizer.py`)

**To the AI Assistant:**
Write the complete code for `visualizer.py`. Use `matplotlib` (Times New Roman, bold axes, 300 DPI). Read the CSV saved by `main.py` to generate the following isolated figures:

* **Figure 1: Agent Convergence Latency (Learning Curve):** * **Y-Axis:** Mean Agent Iterations (averaged across all training scenarios in that epoch).
   * **X-Axis:** Training Epochs.
   * **Plotting:** Plot separate continuous lines for Phase 1 (RIS), Phase 2 (NOMA), and Phase 4 (Joint) training over their respective epochs. 

* **Figure 2: Zero-Shot vs Mastery (The Test Scenarios):** * **Y-Axis:** Absolute Agent Iterations.
   * **X-Axis:** Evaluation Phase (Phase 0 Baseline, Phase 3 Zero-Shot, Phase 4 Final Epoch).
   * **Plotting:** A Bar Chart comparing ONLY the `TEST_JOINT` evaluations across these three distinct milestones.

* **Figure 3: Joint Utility Progression ($U_{JOINT}$):** * **Y-Axis:** $U_{JOINT}$ Score.
   * **X-Axis:** Phase 4 Training Epochs.
   * **Plotting:** A connected line graph tracking the evaluation of `TEST_JOINT` specifically across the epochs of Phase 4. This isolates and proves the fine-tuning capability of the merged database over time.

* **Figure 4: Energy Efficiency Distribution:** * **Y-Axis:** NOMA Power Ratio.
   * **X-Axis:** RIS SNR.
   * **Plotting:** A scatter plot mapping the physical behaviors explored across all phases.

* **Figure 5: Independent Domain Mastery (Learning Curves):** Create a figure with 1x2 subplots (side-by-side) to prove the agent mastered the independent physics over time.
   * **Subplot 1 (RIS):** * **Y-Axis:** $U_{RIS}$ Score.
     * **X-Axis:** Phase 1 Epochs.
     * **Plotting:** A connected line graph tracking the evaluation of `TEST_RIS`. Start at Phase 0 (Baseline) at Epoch 0, and track through all epochs of Phase 1.
   * **Subplot 2 (NOMA):**
     * **Y-Axis:** $U_{NOMA}$ Score.
     * **X-Axis:** Phase 2 Epochs.
     * **Plotting:** A connected line graph tracking the evaluation of `TEST_NOMA`. Start at Phase 0 (Baseline) at Epoch 0, and track through all epochs of Phase 2.