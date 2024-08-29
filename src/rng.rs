const NOISE1: u32 = 0xb5297a4d; // 0b0110'1000'1110'0011'0001'1101'1010'0100
const NOISE2: u32 = 0x68e31da4; // 0b1011'0101'0010'1001'0111'1010'0100'1101
const NOISE3: u32 = 0x1b56c4e9; // 0b0001'1011'0101'0110'1100'0100'1110'1001

fn squirrel3(n: u32, seed: u32) -> u32 {
    let mut n = n;

    n = n.wrapping_mul(NOISE1);
    n = n.wrapping_add(seed);
    n ^= n.wrapping_shr(8);
    n = n.wrapping_add(NOISE2);
    n ^= n.wrapping_shl(8);
    n = n.wrapping_mul(NOISE3);
    n ^= n.wrapping_shr(8);
    n
}

pub struct Rng<T> {
    pos: u32,
    _phantom: std::marker::PhantomData<T>,
}

impl Rng<f32> {
    pub fn new(seed: u32) -> Self {
        Self {
            pos: seed,
            _phantom: std::marker::PhantomData,
        }
    }

    /// returns a random number between 0.0 and 1.0
    pub fn next(&mut self) -> f32 {
        self.pos = squirrel3(self.pos, 0);
        self.pos as f32 / u32::MAX as f32
    }
}
