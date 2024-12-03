use datafusion::common::DFSchemaRef;
use datafusion::logical_expr::{Aggregate, Expr, LogicalPlan, UserDefinedLogicalNodeCore};
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

#[derive(Debug, Eq, PartialEq, PartialOrd, Hash, Clone)]
pub struct GpuAggregatePlanNode {
    pub inner: Aggregate,
}

impl GpuAggregatePlanNode {
    pub fn new(inner: Aggregate) -> Self {
        Self { inner }
    }
    
    pub fn input(&self) -> Arc<LogicalPlan>{
        self.inner.input.clone()
    }
}

impl UserDefinedLogicalNodeCore for GpuAggregatePlanNode {
    fn name(&self) -> &str {
        "GpuAggregate"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        self.inner.input.inputs()
    }

    fn schema(&self) -> &DFSchemaRef {
        &self.inner.schema
    }

    fn expressions(&self) -> Vec<Expr> {
        [self.inner.aggr_expr.clone(), self.inner.group_expr.clone()].concat()
    }

    fn fmt_for_explain(&self, f: &mut Formatter) -> std::fmt::Result {
        self.inner.fmt(f) // TODO
    }

    fn with_exprs_and_inputs(
        &self,
        exprs: Vec<Expr>,
        inputs: Vec<LogicalPlan>,
    ) -> datafusion::common::Result<Self> {
        let mut input = self.inner.input.clone();
        for v in inputs.into_iter() {
            input = Arc::new(v)
        }
        let (aggr, group) = exprs.split_at(self.inner.aggr_expr.len());
        
        Ok(Self { inner: Aggregate::try_new_with_schema(
            input,
            group.to_vec(),
            aggr.to_vec(),
            self.inner.schema.clone(),
        )? })
    }
}
