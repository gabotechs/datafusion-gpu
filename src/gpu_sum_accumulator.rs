use datafusion::arrow::array::ArrayRef;
use datafusion::common::ScalarValue;
use datafusion::physical_plan::Accumulator;

#[derive(Debug)]
pub struct GpuSumAccumulator;

impl Accumulator for GpuSumAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> datafusion::common::Result<()> {
        todo!()
    }

    fn evaluate(&mut self) -> datafusion::common::Result<ScalarValue> {
        todo!()
    }

    fn size(&self) -> usize {
        todo!()
    }

    fn state(&mut self) -> datafusion::common::Result<Vec<ScalarValue>> {
        todo!()
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> datafusion::common::Result<()> {
        todo!()
    }
}
