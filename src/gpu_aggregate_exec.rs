use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::aggregate::AggregateExprBuilder;
use datafusion::physical_plan::aggregates::AggregateExec;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use delegate::delegate;
use std::any::Any;
use std::fmt::Formatter;
use std::sync::Arc;

impl TryFrom<&AggregateExec> for GpuAggregateExec {
    type Error = datafusion::error::DataFusionError;

    fn try_from(value: &AggregateExec) -> Result<Self, Self::Error> {
        let mut v = vec![];
        for agg in value.aggr_expr() {
            // TODO: perform some conversions
            v.push(agg.clone());
        }
        let inner = value.with_new_aggr_exprs(v);
        Ok(Self {
            inner: Arc::new(inner),
        })
    }
}

#[derive(Debug)]
pub struct GpuAggregateExec {
    pub inner: Arc<AggregateExec>,
}

impl DisplayAs for GpuAggregateExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "GpuAggregateExec")
    }
}

impl ExecutionPlan for GpuAggregateExec {
    fn name(&self) -> &str {
        "GpuAggregateExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    delegate! {
        to self.inner {
            fn properties(&self) -> &PlanProperties;
            fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>>;
        }
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        let inner = self.inner.clone().with_new_children(children)?;
        // TODO: is there a less-overhead way of doing this?
        let inner = inner
            .as_any()
            .downcast_ref::<AggregateExec>()
            .ok_or(datafusion::common::DataFusionError::Internal("When calling GpuAggregateExec.with_new_children, the inner node could not be casted as an AggregateExec node".into()))?;
        Ok(Arc::new(Self {
            inner: Arc::new(inner.clone()),
        }))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> datafusion::common::Result<SendableRecordBatchStream> {
        // TODO: custom execution
        self.inner.execute(partition, context)
    }
}
