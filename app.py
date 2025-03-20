import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from heapq import heappush, heappop

st.title("LSTF Task Scheduling Simulator")

tab1, tab2, tab3 = st.tabs(["📂 Input Tasks", "📊 Scheduling Results", "📜 Summary"])

# ---- TASK INPUT ----
with tab1:
    st.subheader("Enter Task Details")
    num_tasks = st.number_input("Enter number of tasks:", min_value=2, step=1, value=3)

    tasks = {}
    data_volumes = {}  # Store data volume (lambda) for each dependency
    for i in range(num_tasks):
        with st.expander(f"Task {i+1} Details"):
            execution_time = st.number_input(f"Execution time of Task {i+1}:", min_value=1, key=f"et_{i}")
            deadline = st.number_input(f"Deadline of Task {i+1}:", min_value=1, key=f"dl_{i}")
            dependencies = st.text_input(f"Dependencies (comma-separated task numbers) for Task {i+1}:", key=f"dep_{i}")
            dep_volumes = st.text_input(f"Data volumes for dependencies (comma-separated, matching order), leave blank for 1:", 
                                       key=f"vol_{i}")

            try:
                dependencies = list(map(int, dependencies.split(','))) if dependencies else []
                dependencies = [d for d in dependencies if 1 <= d <= num_tasks and d != i+1]
            except ValueError:
                dependencies = []

            try:
                volumes = list(map(int, dep_volumes.split(','))) if dep_volumes else [1] * len(dependencies)
                if len(volumes) != len(dependencies):
                    st.warning(f"Task {i+1}: Number of data volumes must match number of dependencies. Using 1 for all.")
                    volumes = [1] * len(dependencies)
            except ValueError:
                volumes = [1] * len(dependencies)

            tasks[i+1] = {
                "execution_time": execution_time,
                "deadline": deadline,
                "dependencies": dependencies
            }
            for dep, vol in zip(dependencies, volumes):
                data_volumes[(dep, i+1)] = vol

    num_processors = st.number_input("Enter number of processors:", min_value=1, step=1, value=2)

# ---- COMMUNICATION COST MATRIX INPUT ----
st.subheader("Enter Communication Cost Matrix (Task-to-Task)")
matrix_option = st.radio("How do you want to enter the communication cost matrix?", ["Manual Entry", "Auto-Generate"], index=0)

communication_costs = None

if matrix_option == "Manual Entry":
    matrix_input = st.text_area(f"Enter {num_tasks}x{num_tasks} matrix (space-separated rows, task-to-task costs per unit data)", 
                               value="0 1 2\n1 0 3\n2 3 0" if num_tasks == 3 else "0 1\n1 0", height=150)
    try:
        rows = matrix_input.strip().split("\n")
        communication_costs = np.array([list(map(int, row.split())) for row in rows])
        if communication_costs.shape != (num_tasks, num_tasks):
            st.error(f"Invalid matrix size! Expected {num_tasks}x{num_tasks} for {num_tasks} tasks.")
            communication_costs = None
    except ValueError:
        st.error("Invalid input! Please enter only integers.")
        communication_costs = None
elif matrix_option == "Auto-Generate":
    max_cost = st.slider("Max Communication Cost per Unit Data", min_value=1, max_value=20, value=5)
    np.random.seed(42)
    communication_costs = np.random.randint(1, max_cost, size=(num_tasks, num_tasks))
    np.fill_diagonal(communication_costs, 0)
    communication_costs = (communication_costs + communication_costs.T) // 2
    st.write("Generated Communication Cost Matrix (Task-to-Task):")
    st.write(pd.DataFrame(communication_costs, 
                         index=[f"T{i+1}" for i in range(num_tasks)], 
                         columns=[f"T{i+1}" for i in range(num_tasks)]))

# ---- TASK DEPENDENCY GRAPH ----
st.subheader("Task Dependency Graph (Original)")
if tasks:
    G = nx.DiGraph()
    for task, details in tasks.items():
        G.add_node(task, label=f"Task {task} (D={details['deadline']})")
        for dep in details["dependencies"]:
            G.add_edge(dep, task, volume=data_volumes.get((dep, task), 1))

    # Check if graph is acyclic
    if not nx.is_directed_acyclic_graph(G):
        st.error("The task dependency graph contains cycles! LSTF requires a Directed Acyclic Graph (DAG).")
    else:
        pos = nx.spring_layout(G, seed=42)
        fig = go.Figure()

        for u, v in G.edges():
            fig.add_trace(go.Scatter(x=[pos[u][0], pos[v][0]], y=[pos[u][1], pos[v][1]],
                                    mode='lines', line=dict(width=1, color="gray"),
                                    hoverinfo='text', text=f"{u} → {v} (λ={G[u][v]['volume']})"))

        for node in G.nodes():
            fig.add_trace(go.Scatter(x=[pos[node][0]], y=[pos[node][1]], mode='markers+text',
                                    marker=dict(size=15, color="blue"),
                                    text=G.nodes[node]['label'], textposition="top center"))

        fig.update_layout(title="Task Dependencies (Original Deadlines)", 
                         xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
        st.plotly_chart(fig, use_container_width=True)

# ---- LSTF SCHEDULING FUNCTIONS ----
def compute_modified_deadlines(tasks):
    """Compute modified deadlines based on precedence constraints."""
    modified_deadlines = {task: details["deadline"] for task, details in tasks.items()}
    for task in tasks:
        for succ in [t for t, d in tasks.items() if task in d["dependencies"]]:
            modified_deadlines[task] = min(modified_deadlines[task], 
                                          modified_deadlines[succ] - tasks[succ]["execution_time"])
    return modified_deadlines

def compute_space_time(tasks, modified_deadlines):
    """Calculate space-time for each task."""
    return {task: modified_deadlines[task] - details["execution_time"] 
            for task, details in tasks.items()}

def schedule_lstf(tasks, num_processors, communication_costs, data_volumes):
    """Implement LSTF with task-to-task communication costs scaled by data volume."""
    modified_deadlines = compute_modified_deadlines(tasks)
    space_time = compute_space_time(tasks, modified_deadlines)
    
    ready_queue = []
    scheduled = set()
    schedule = []
    processor_end_times = [0] * num_processors
    task_assignment = {}
    logs = []
    missed_deadlines = 0
    tardiness = 0
    tardiness_per_task = {}

    # Initialize with tasks that have no dependencies
    for task in tasks:
        if not tasks[task]["dependencies"]:
            heappush(ready_queue, (space_time[task], task))

    while ready_queue:
        _, task = heappop(ready_queue)
        if task in scheduled:
            continue

        # Find earliest start processor
        earliest_start = float('inf')
        best_processor = 0
        for p in range(num_processors):
            start_time = processor_end_times[p]
            for dep in tasks[task]["dependencies"]:
                if dep in task_assignment:
                    prev_proc = task_assignment[dep]
                    comm_cost = communication_costs[dep-1][task-1] * data_volumes.get((dep, task), 1)
                    comm_delay = comm_cost if prev_proc != p else 0
                    dep_end = next(s[3] for s in schedule if s[0] == dep)
                    start_time = max(start_time, dep_end + comm_delay)
            if start_time < earliest_start:
                earliest_start = start_time
                best_processor = p

        end_time = earliest_start + tasks[task]["execution_time"]
        schedule.append((task, best_processor, earliest_start, end_time))
        processor_end_times[best_processor] = end_time
        task_assignment[task] = best_processor
        scheduled.add(task)
        logs.append(f"Task {task} → Processor {best_processor}, Start: {earliest_start}, End: {end_time}")
        
        task_tardiness = max(0, end_time - tasks[task]["deadline"])
        tardiness_per_task[task] = task_tardiness
        if task_tardiness > 0:
            missed_deadlines += 1
            tardiness = max(tardiness, task_tardiness)

        # Add newly ready tasks
        for succ in [t for t, d in tasks.items() if task in d["dependencies"]]:
            if all(dep in scheduled for dep in tasks[succ]["dependencies"]) and succ not in scheduled:
                heappush(ready_queue, (space_time[succ], succ))

    total_execution_time = max(processor_end_times)
    return schedule, logs, total_execution_time, missed_deadlines, tardiness, modified_deadlines, tardiness_per_task

# ---- RUN SIMULATION ----
if st.button("Run Simulation") and communication_costs is not None and nx.is_directed_acyclic_graph(G):
    with tab2:
        st.subheader("LSTF Scheduling Simulation")
        schedule, logs, total_execution_time, missed_deadlines, tardiness, modified_deadlines, tardiness_per_task = schedule_lstf(
            tasks, num_processors, communication_costs, data_volumes)
        
        if schedule:
            # Scheduling Table
            df_schedule = pd.DataFrame(schedule, columns=["Task", "Processor", "Start Time", "End Time"])
            st.write("### 🔹 Scheduling Result")
            st.dataframe(df_schedule)

            # Gantt Chart
            st.write("### 🔹 Gantt Chart")
            fig_gantt = go.Figure()
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Add more if needed
            for i, (task, proc, start, end) in enumerate(schedule):
                fig_gantt.add_trace(go.Bar(
                    x=[end - start], y=[f"Processor {proc}"],
                    base=[start], orientation='h',
                    marker=dict(color=colors[i % len(colors)]),
                    name=f"Task {task}",
                    hoverinfo='text',
                    text=f"Task {task}: {start} - {end}"
                ))
            fig_gantt.update_layout(
                title="Task Schedule (Gantt Chart)",
                xaxis_title="Time",
                yaxis_title="Processor",
                barmode='stack',
                showlegend=True
            )
            st.plotly_chart(fig_gantt, use_container_width=True)

            # Modified Dependency Graph
            st.write("### 🔹 Task Dependency Graph (Modified Deadlines)")
            G_mod = nx.DiGraph()
            for task, details in tasks.items():
                G_mod.add_node(task, label=f"Task {task} (D={modified_deadlines[task]})")
                for dep in details["dependencies"]:
                    G_mod.add_edge(dep, task, volume=data_volumes.get((dep, task), 1))

            pos_mod = nx.spring_layout(G_mod, seed=42)
            fig_mod = go.Figure()
            for u, v in G_mod.edges():
                fig_mod.add_trace(go.Scatter(x=[pos_mod[u][0], pos_mod[v][0]], y=[pos_mod[u][1], pos_mod[v][1]],
                                            mode='lines', line=dict(width=1, color="gray"),
                                            hoverinfo='text', text=f"{u} → {v} (λ={G_mod[u][v]['volume']})"))
            for node in G_mod.nodes():
                fig_mod.add_trace(go.Scatter(x=[pos_mod[node][0]], y=[pos_mod[node][1]], mode='markers+text',
                                            marker=dict(size=15, color="green"),
                                            text=G_mod.nodes[node]['label'], textposition="top center"))
            fig_mod.update_layout(title="Task Dependencies (Modified Deadlines)", 
                                 xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
            st.plotly_chart(fig_mod, use_container_width=True)

            # Logs and Metrics
            st.write("### 🔹 Execution Log")
            st.text("\n".join(logs))
            st.write(f"### 🔹 Total Execution Time: {total_execution_time}")
            st.write(f"### 🔹 Missed Deadlines: {missed_deadlines}")
            st.write(f"### 🔹 Maximum Tardiness: {tardiness}")

    with tab3:
        st.subheader("Simulation Results Summary")
        if schedule:
            # Basic Metrics
            st.write(f"**Total Tasks**: {num_tasks}")
            st.write(f"**Number of Processors**: {num_processors}")
            st.write(f"**Total Execution Time**: {total_execution_time}")
            st.write(f"**Tasks with Missed Deadlines**: {missed_deadlines} ({(missed_deadlines/num_tasks)*100:.1f}%)")
            st.write(f"**Maximum Tardiness**: {tardiness}")

            # Average Tardiness
            avg_tardiness = sum(tardiness_per_task.values()) / num_tasks
            st.write(f"**Average Tardiness per Task**: {avg_tardiness:.2f}")

            # Tasks per Processor
            processor_load = {p: [] for p in range(num_processors)}
            for task, proc, start, end in schedule:
                processor_load[proc].append(task)
            st.write("**Task Distribution Across Processors**:")
            for p, task_list in processor_load.items():
                st.write(f"- Processor {p}: {len(task_list)} tasks ({', '.join(map(str, task_list))})")

            # Deadline Performance
            st.write("**Deadline Performance**:")
            for task in tasks:
                original_deadline = tasks[task]["deadline"]
                modified_deadline = modified_deadlines[task]
                end_time = next(s[3] for s in schedule if s[0] == task)
                tardiness_val = tardiness_per_task[task]
                status = "Met" if tardiness_val == 0 else "Missed"
                st.write(f"- Task {task}: Original D={original_deadline}, Modified D={modified_deadline}, "
                         f"Completed={end_time}, Tardiness={tardiness_val}, Status={status}")

            # Performance Insight
            if missed_deadlines == 0:
                st.success("All deadlines were met!")
            elif missed_deadlines < num_tasks / 2:
                st.warning(f"Some deadlines missed ({missed_deadlines}/{num_tasks}). Consider more processors or relaxed deadlines.")
            else:
                st.error(f"Most deadlines missed ({missed_deadlines}/{num_tasks}). System may be overloaded.")
        else:
            st.write("No simulation results available. Please run the simulation in the 'Scheduling Results' tab.")