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
    let mut xb = vec![Uniform::from(0.0 .. 1.0).sample(&mut rng) as f32; d * nb];
    // manipulate the first element of each entry for this experiment
    for i in 0..nb {
        xb[i*d] += i as f32 / manipulation_v;
    }
    
    // initialize uniformly distributed query data
    let mut xq = vec![Uniform::from(0.0 .. 1.0).sample(&mut rng) as f32; d * nq];
    // manipulate..
    for i in 0..nq {
        xq[i*d] += i as f32 / manipulation_v;
    }

    // build faiss index
    let mut index = index_factory(d as u32, "Flat", MetricType::L2).expect("Failed to initiaite an index");
    println!("{:?}", index.is_trained());
    let _ = index.add(&xb);
    println!("{:?}", index.ntotal());
    let _ = index.train(&xb); // this is unnecessary for this example but why not.

    // number of neighbors to find per query
    let k = 4;

    // sanity check
    println!("{}", "<SANITY CHECK>");
    let result = index.search(&xb[0..5*d], k).expect("Failed to search.");
    
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
