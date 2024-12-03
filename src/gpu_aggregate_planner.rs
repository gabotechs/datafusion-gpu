use crate::gpu_aggregate_exec::GpuAggregateExec;
use crate::gpu_aggregate_plan_node::GpuAggregatePlanNode;
use async_trait::async_trait;
use datafusion::execution::SessionState;
use datafusion::logical_expr::{Extension, LogicalPlan, UserDefinedLogicalNode};
use datafusion::physical_plan::aggregates::AggregateExec;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_planner::{ExtensionPlanner, PhysicalPlanner};
use std::sync::Arc;

pub struct GpuAggregatePlanner;

#[async_trait]
impl ExtensionPlanner for GpuAggregatePlanner {
    async fn plan_extension(
        &self,
        planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        session_state: &SessionState,
    ) -> datafusion::common::Result<Option<Arc<dyn ExecutionPlan>>> {
        Ok(
            if let Some(custom_node) = node.as_any().downcast_ref::<GpuAggregatePlanNode>() {
                let plan = planner
                    .create_physical_plan(
                        &LogicalPlan::Aggregate(custom_node.inner.clone()),
                        session_state,
                    )
                    .await?;
                if let Some(agg_plan) = plan.as_any().downcast_ref::<AggregateExec>() {
                    Some(Arc::new(GpuAggregateExec::try_from(agg_plan)?))
                } else {
                    None
                }
            } else {
                None
            },
        )
    }
}
