import React, { useState } from "react";
import {
  CheckCircle2,
  Play,
  XCircle,
  Plus,
  UserCheck,
  Calendar,
  ShieldAlert,
} from "lucide-react";
import { apiClient } from "../api/client";
import type { TaskItem, TaskPriority, TaskStatus } from "../types";

interface ActionPlanBoardProps {
  dischargeId: number;
  initialTasks: TaskItem[];
  onTaskUpdated?: () => void;
  readOnly?: boolean;
}

const COORDINATOR_ROLES = [
  "Care Coordinator",
  "Nurse Navigator",
  "Clinical Pharmacist",
  "Social Worker",
  "Transition Coach",
  "Attending Physician",
];

export const ActionPlanBoard: React.FC<ActionPlanBoardProps> = ({
  dischargeId,
  initialTasks,
  onTaskUpdated,
  readOnly = false,
}) => {
  const [tasks, setTasks] = useState<TaskItem[]>(initialTasks || []);
  const [filterStatus, setFilterStatus] = useState<string>("all");
  const [isAddingTask, setIsAddingTask] = useState<boolean>(false);
  const [updatingTaskId, setUpdatingTaskId] = useState<number | null>(null);

  // New task form state
  const [newTitle, setNewTitle] = useState("");
  const [newCategory, setNewCategory] = useState("Care Transition");
  const [newPriority, setNewPriority] = useState<TaskPriority>("High");
  const [newRationale, setNewRationale] = useState("");
  const [newTriggerDriver, setNewTriggerDriver] = useState("");
  const [newDueDate, setNewDueDate] = useState("");
  const [newOwner, setNewOwner] = useState("Care Coordinator");

  // Keep local tasks in sync if initialTasks change
  React.useEffect(() => {
    setTasks(initialTasks || []);
  }, [initialTasks]);

  const handleUpdateStatus = async (
    taskId: number,
    status: TaskStatus,
    note?: string
  ) => {
    try {
      setUpdatingTaskId(taskId);
      await apiClient.updateTask(dischargeId, taskId, {
        status,
        completion_note: note,
        actor: "Care Coordinator",
      });
      // Refresh local tasks from API
      const updated = await apiClient.getTasks(dischargeId);
      setTasks(updated);
      onTaskUpdated?.();
    } catch (err: any) {
      alert(`Error updating task status: ${err.message}`);
    } finally {
      setUpdatingTaskId(null);
    }
  };

  const handleUpdateOwner = async (taskId: number, owner: string) => {
    try {
      setUpdatingTaskId(taskId);
      await apiClient.updateTask(dischargeId, taskId, {
        owner,
        actor: "Care Coordinator",
      });
      const updated = await apiClient.getTasks(dischargeId);
      setTasks(updated);
      onTaskUpdated?.();
    } catch (err: any) {
      alert(`Error reassigning task: ${err.message}`);
    } finally {
      setUpdatingTaskId(null);
    }
  };

  const handleCreateTask = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newTitle.trim()) return;
    try {
      await apiClient.createTask(dischargeId, {
        title: newTitle.trim(),
        category: newCategory,
        priority: newPriority,
        rationale: newRationale.trim() || "Post-discharge care coordination task",
        trigger_driver: newTriggerDriver.trim() || undefined,
        due_date: newDueDate || undefined,
        owner: newOwner,
        actor: "Care Coordinator",
      });
      // Reset form
      setNewTitle("");
      setNewRationale("");
      setNewTriggerDriver("");
      setNewDueDate("");
      setIsAddingTask(false);
      const updated = await apiClient.getTasks(dischargeId);
      setTasks(updated);
      onTaskUpdated?.();
    } catch (err: any) {
      alert(`Error creating task: ${err.message}`);
    }
  };

  const filteredTasks = tasks.filter((t) => {
    if (filterStatus === "all") return true;
    return t.status === filterStatus;
  });

  const completedCount = tasks.filter((t) => t.status === "completed").length;
  const pendingCount = tasks.filter((t) => t.status === "pending").length;
  const inProgressCount = tasks.filter((t) => t.status === "in_progress").length;

  return (
    <div className="space-y-4">
      {/* Stage Header & Decision Support Banner */}
      <div className="bg-canvas hairline-border rounded-sm p-4 space-y-2">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-hairline pb-2">
          <div>
            <div className="flex items-center gap-2">
              <span className="font-mono text-[10px] uppercase px-1.5 py-0.5 bg-accent/10 text-accent font-bold rounded">
                ActionLoop Stage 3
              </span>
              <h3 className="text-sm font-semibold text-primary">
                Post-Discharge Action Plan & Care Protocol
              </h3>
            </div>
            <p className="text-xs text-muted mt-0.5">
              Deterministic rule-based follow-up tasks derived from clinical risk drivers.
            </p>
          </div>

          <div className="flex items-center gap-2 font-mono text-xs">
            <span className="px-2 py-0.5 bg-protective/10 text-protective rounded font-medium">
              {completedCount} Done
            </span>
            <span className="px-2 py-0.5 bg-accent/10 text-accent rounded font-medium">
              {inProgressCount} Active
            </span>
            <span className="px-2 py-0.5 bg-subtle text-muted rounded font-medium">
              {pendingCount} Pending
            </span>
          </div>
        </div>

        {/* Clinician Decision Support Mandatory Label */}
        <div className="flex items-start gap-2 p-2.5 bg-caution/10 border border-caution/30 rounded text-xs text-caution">
          <ShieldAlert className="w-4 h-4 shrink-0 mt-0.5 text-caution" />
          <div className="text-[11px] leading-relaxed">
            <strong>Clinical Decision Support Notice:</strong> Suggested interventions are rule-based recommendations generated from patient risk factors. They do not constitute autonomous clinical orders and require review and authorization by an attending physician or licensed care coordinator.
          </div>
        </div>
      </div>

      {/* Filter and New Task Bar */}
      <div className="flex items-center justify-between gap-3 text-xs">
        <div className="flex items-center gap-1.5">
          <span className="text-muted font-mono uppercase text-[10px]">Filter:</span>
          {["all", "pending", "in_progress", "completed"].map((st) => (
            <button
              key={st}
              onClick={() => setFilterStatus(st)}
              className={`px-2 py-1 rounded-sm font-mono text-[11px] capitalize transition-colors ${
                filterStatus === st
                  ? "bg-accent text-white font-semibold"
                  : "bg-surface hover:bg-subtle text-muted hairline-border"
              }`}
            >
              {st.replace("_", " ")}
            </button>
          ))}
        </div>

        {!readOnly && (
          <button
            onClick={() => setIsAddingTask(!isAddingTask)}
            className="px-3 py-1 bg-surface hover:bg-subtle hairline-border text-primary font-medium rounded-sm flex items-center gap-1.5 transition-colors"
          >
            <Plus className="w-3.5 h-3.5 text-accent" />
            <span>{isAddingTask ? "Cancel" : "Add Follow-up Task"}</span>
          </button>
        )}
      </div>

      {/* New Task Inline Form */}
      {isAddingTask && (
        <form
          onSubmit={handleCreateTask}
          className="p-4 bg-surface hairline-border rounded-sm space-y-3 text-xs animate-fade-in"
        >
          <div className="font-semibold text-primary border-b border-hairline pb-1.5">
            Create Custom Follow-up Care Task
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div>
              <label className="block text-muted mb-1 font-mono text-[11px]">
                Task Title *
              </label>
              <input
                type="text"
                required
                placeholder="e.g. In-home post-discharge medication check"
                value={newTitle}
                onChange={(e) => setNewTitle(e.target.value)}
                className="w-full bg-canvas hairline-border rounded px-2.5 py-1.5 text-primary"
              />
            </div>

            <div>
              <label className="block text-muted mb-1 font-mono text-[11px]">
                Clinical Category
              </label>
              <select
                value={newCategory}
                onChange={(e) => setNewCategory(e.target.value)}
                className="w-full bg-canvas hairline-border rounded px-2.5 py-1.5 text-primary"
              >
                <option value="Care Transition">Care Transition</option>
                <option value="Medication Management">Medication Management</option>
                <option value="Clinical Follow-Up">Clinical Follow-Up</option>
                <option value="Social Determinants">Social Determinants / DME</option>
                <option value="Patient Education">Patient Education</option>
              </select>
            </div>

            <div>
              <label className="block text-muted mb-1 font-mono text-[11px]">
                Priority Tier
              </label>
              <select
                value={newPriority}
                onChange={(e) => setNewPriority(e.target.value as TaskPriority)}
                className="w-full bg-canvas hairline-border rounded px-2.5 py-1.5 text-primary"
              >
                <option value="High">High</option>
                <option value="Medium">Medium</option>
                <option value="Low">Low</option>
              </select>
            </div>

            <div>
              <label className="block text-muted mb-1 font-mono text-[11px]">
                Assigned Owner
              </label>
              <select
                value={newOwner}
                onChange={(e) => setNewOwner(e.target.value)}
                className="w-full bg-canvas hairline-border rounded px-2.5 py-1.5 text-primary"
              >
                {COORDINATOR_ROLES.map((role) => (
                  <option key={role} value={role}>
                    {role}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-muted mb-1 font-mono text-[11px]">
                Target Due Date
              </label>
              <input
                type="date"
                value={newDueDate}
                onChange={(e) => setNewDueDate(e.target.value)}
                className="w-full bg-canvas hairline-border rounded px-2.5 py-1.5 text-primary"
              />
            </div>

            <div>
              <label className="block text-muted mb-1 font-mono text-[11px]">
                Trigger Risk Driver (Optional)
              </label>
              <input
                type="text"
                placeholder="e.g. High medication regimen changes"
                value={newTriggerDriver}
                onChange={(e) => setNewTriggerDriver(e.target.value)}
                className="w-full bg-canvas hairline-border rounded px-2.5 py-1.5 text-primary"
              />
            </div>
          </div>

          <div>
            <label className="block text-muted mb-1 font-mono text-[11px]">
              Clinical Rationale
            </label>
            <input
              type="text"
              placeholder="Why this outreach intervention is recommended for this discharge"
              value={newRationale}
              onChange={(e) => setNewRationale(e.target.value)}
              className="w-full bg-canvas hairline-border rounded px-2.5 py-1.5 text-primary"
            />
          </div>

          <div className="flex justify-end gap-2 pt-2 border-t border-hairline">
            <button
              type="button"
              onClick={() => setIsAddingTask(false)}
              className="px-3 py-1.5 bg-canvas hairline-border text-muted rounded hover:text-primary"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-1.5 bg-accent text-white font-medium rounded hover:opacity-90"
            >
              Save Follow-up Task
            </button>
          </div>
        </form>
      )}

      {/* Task Cards Grid */}
      <div className="space-y-3">
        {filteredTasks.length === 0 ? (
          <div className="p-6 bg-canvas hairline-border rounded-sm text-center text-xs text-muted font-mono">
            No care transition tasks match the selected filter.
          </div>
        ) : (
          filteredTasks.map((task) => {
            const isCompleted = task.status === "completed";
            const isInProgress = task.status === "in_progress";
            const isCancelled = task.status === "cancelled";
            const isPending = task.status === "pending";

            return (
              <div
                key={task.id}
                className={`p-4 rounded-sm hairline-border transition-all ${
                  isCompleted
                    ? "bg-canvas/60 border-l-4 border-l-protective opacity-90"
                    : isInProgress
                    ? "bg-surface border-l-4 border-l-accent shadow-xs"
                    : isCancelled
                    ? "bg-canvas border-l-4 border-l-muted opacity-60"
                    : "bg-surface border-l-4 border-l-caution"
                }`}
              >
                <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-2">
                  <div className="space-y-1.5 flex-1">
                    {/* Tags row */}
                    <div className="flex flex-wrap items-center gap-2 text-[10px] font-mono">
                      <span
                        className={`px-1.5 py-0.5 rounded font-bold uppercase ${
                          task.priority === "High"
                            ? "bg-critical/10 text-critical"
                            : task.priority === "Medium"
                            ? "bg-caution/10 text-caution"
                            : "bg-protective/10 text-protective"
                        }`}
                      >
                        {task.priority} Priority
                      </span>

                      <span className="px-1.5 py-0.5 bg-subtle text-muted rounded uppercase">
                        {task.category}
                      </span>

                      {task.trigger_driver && (
                        <span className="px-1.5 py-0.5 bg-accent/10 text-accent rounded font-medium">
                          ⚡ Trigger: {task.trigger_driver}
                        </span>
                      )}
                    </div>

                    {/* Title */}
                    <h4
                      className={`text-xs font-semibold ${
                        isCompleted ? "line-through text-muted" : "text-primary"
                      }`}
                    >
                      {task.title}
                    </h4>

                    {/* Rationale */}
                    {task.rationale && (
                      <p className="text-[11px] text-muted leading-relaxed">
                        {task.rationale}
                      </p>
                    )}

                    {/* Meta: due date & owner */}
                    <div className="flex flex-wrap items-center gap-4 text-[11px] text-muted pt-1">
                      {task.due_date && (
                        <div className="flex items-center gap-1 font-mono">
                          <Calendar className="w-3 h-3 text-muted" />
                          <span>Due: {task.due_date}</span>
                        </div>
                      )}

                      <div className="flex items-center gap-1 font-mono">
                        <UserCheck className="w-3 h-3 text-muted" />
                        <span>Owner:</span>
                        {!readOnly ? (
                          <select
                            value={task.owner || "Care Coordinator"}
                            onChange={(e) =>
                              handleUpdateOwner(task.id, e.target.value)
                            }
                            disabled={updatingTaskId === task.id}
                            className="bg-canvas hairline-border rounded px-1.5 py-0.5 text-primary text-[10px]"
                          >
                            {COORDINATOR_ROLES.map((r) => (
                              <option key={r} value={r}>
                                {r}
                              </option>
                            ))}
                          </select>
                        ) : (
                          <span className="text-primary font-medium">
                            {task.owner || "Unassigned"}
                          </span>
                        )}
                      </div>

                      {task.completed_at && (
                        <span className="text-protective text-[10px] font-mono">
                          ✓ Completed by {task.completed_by || "Coordinator"} on{" "}
                          {new Date(task.completed_at).toLocaleDateString()}
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Status Action Buttons */}
                  {!readOnly && (
                    <div className="flex items-center gap-1.5 shrink-0 pt-2 sm:pt-0">
                      {isPending && (
                        <button
                          onClick={() =>
                            handleUpdateStatus(task.id, "in_progress")
                          }
                          disabled={updatingTaskId === task.id}
                          className="px-2.5 py-1 bg-accent text-white rounded text-[11px] font-medium flex items-center gap-1 hover:opacity-90"
                        >
                          <Play className="w-3 h-3" />
                          <span>Start</span>
                        </button>
                      )}

                      {(isPending || isInProgress) && (
                        <button
                          onClick={() => {
                            const note = prompt(
                              "Completion touchpoint note (e.g. Patient contacted, medication schedule confirmed):"
                            );
                            handleUpdateStatus(
                              task.id,
                              "completed",
                              note || undefined
                            );
                          }}
                          disabled={updatingTaskId === task.id}
                          className="px-2.5 py-1 bg-protective text-white rounded text-[11px] font-medium flex items-center gap-1 hover:opacity-90"
                        >
                          <CheckCircle2 className="w-3 h-3" />
                          <span>Complete</span>
                        </button>
                      )}

                      {isCompleted && (
                        <button
                          onClick={() =>
                            handleUpdateStatus(task.id, "in_progress")
                          }
                          disabled={updatingTaskId === task.id}
                          className="px-2 py-1 bg-surface hairline-border hover:bg-subtle text-muted rounded text-[10px] font-mono"
                        >
                          Reopen
                        </button>
                      )}

                      {!isCancelled && !isCompleted && (
                        <button
                          onClick={() =>
                            handleUpdateStatus(task.id, "cancelled")
                          }
                          disabled={updatingTaskId === task.id}
                          className="px-2 py-1 bg-canvas hover:bg-subtle text-muted hover:text-critical rounded text-[10px] font-mono"
                          title="Cancel Task"
                        >
                          <XCircle className="w-3.5 h-3.5" />
                        </button>
                      )}
                    </div>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
