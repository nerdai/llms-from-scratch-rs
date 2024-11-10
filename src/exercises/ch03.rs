use crate::Exercise;

/// 3.1
pub struct X3P1;

impl Exercise for X3P1 {
    fn name(&self) -> String {
        String::from("3.1")
    }

    fn main(&self) {
        use crate::listings::ch03::{SelfAttentionV1, SelfAttentionV2};
        use candle_core::Device;
        todo!()
    }
}
