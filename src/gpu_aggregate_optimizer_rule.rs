use crate::gpu_aggregate_plan_node::GpuAggregatePlanNode;
use datafusion::common::tree_node::Transformed;
use datafusion::common::DataFusionError;
use datafusion::logical_expr::{Extension, LogicalPlan};
use datafusion::optimizer::optimizer::ApplyOrder;
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};
use std::sync::Arc;

#[derive(Debug)]
pub struct GPUAggregateOptimizerRule;

impl OptimizerRule for GPUAggregateOptimizerRule {
    fn name(&self) -> &str {
        "GpuAggregateOptimizerRule"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::TopDown)
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> datafusion::common::Result<Transformed<LogicalPlan>, DataFusionError> {
        if let LogicalPlan::Aggregate(agg) = plan {
            Ok(Transformed::yes(LogicalPlan::Extension(Extension {
                node: Arc::new(GpuAggregatePlanNode::new(agg)),
            })))
        } else {
            Ok(Transformed::no(plan))
        }
    }
}
