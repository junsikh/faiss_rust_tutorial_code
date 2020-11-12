use faiss::{Index, index_factory, MetricType};
use rand::{SeedableRng, rngs::StdRng};
use rand::distributions::{Distribution, Uniform};


fn main() {
    println!("faiss tutorial test");
    // Reference in Python:
    // https://github.com/facebookresearch/faiss/blob/master/tutorial/python/1-Flat.py

    // prepare data
    let d = 64; // dimension
    let nb = 100000; // database size
    let nq = 5; // nb of queries
    let seed = 1234; // make reproducible
    let mut rng = StdRng::seed_from_u64(seed);

    let manipulation_v: f32 = 1000.0;
    
    // initialize uniformly distributed data
    let mut xb: Vec<f32> = Vec::with_capacity(d * nb);
    for _ in 0..xb.capacity() {
        // let v: f32 = StandardNormal.sample(&mut rng) as f32;
        let v = Uniform::from(0.0 .. 1.0).sample(&mut rng) as f32;
        xb.push(v)
    }
    // manipulate the first element of each entry for this experiment
    for i in 0..nb {
        xb[i*d] += i as f32 / manipulation_v;
    }
    
    // initialize uniformly distributed query data
    let mut xq: Vec<f32> = Vec::with_capacity(d * nq);
    for _ in 0..xq.capacity() {
        // let v: f32 = StandardNormal.sample(&mut rng) as f32;
        let v = Uniform::from(0.0 .. 1.0).sample(&mut rng) as f32;
        xq.push(v)
    }
    // manipulate..
    for i in 0..nq {
        xq[i*d] += i as f32 / manipulation_v;
    }

    // build faiss index
    let mut index = match index_factory(d as u32, "Flat", MetricType::L2) {
        Ok(index) => index,
        Err(error) => panic!("Failed to initiaite an index. {:?}", error)
    };
    println!("{:?}", index.is_trained());
    index.add(&xb);
    println!("{:?}", index.ntotal());
    index.train(&xb); // this is unnecessary for this example but why not.

    // number of neighbors to find per query
    let k = 4;

    // sanity check
    println!("{}", "<SANITY CHECK>");
    let result = match index.search(&xb[0..5*d], 4) {
        Ok(res) => res,
        Err(error) => panic!("Failed to search. {:?}", error)
    };
    for (i, (l, d)) in result.labels.iter()
        .zip(result.distances.iter())
        .enumerate()
    {
        println!("#{}: {} (D={})", i+1, *l, *d);
    }
    println!("\n");

    // search test
    println!("{}", "<SEARCH TEST>");
    let result = match index.search(&xq, 4) {
        Ok(res) => res,
        Err(error) => panic!("Failed to search. {:?}", error)
    };
    for (i, (l, d)) in result.labels.iter()
        .zip(result.distances.iter())
        .enumerate()
    {
        println!("#{}: {} (D={})", i+1, *l, *d);
    }
    println!("\n");

}
