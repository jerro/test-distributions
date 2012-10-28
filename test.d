import std.stdio, std.mathspecial, std.algorithm, std.range, std.conv, 
    std.traits, std.numeric, std.random, std.typecons, std.typetuple,
    std.parallelism, std.getopt;

import dstats.all;

mixin template NormalMixin()
{
    T pdfMax()
    {
        return 1 / (sigma * sqrt(2 * PI));
    } 

    T cdf(T x)
    {
        return normalCDF(x, mean, sigma); 
    } 
}

struct DstatsNormalTest(T)
{
    T mean = 0;
    T sigma = 1;

    T sample(Rng)(ref Rng rng) @system
    {
        return rNorm(mean, sigma, rng);
    }
    
    mixin NormalMixin!();
}

struct NormalDistTest(T, size_t nlayers)
{
    enum T mean = 0;
    enum T sigma = 1;
    NormalDist!(T, nlayers) dist;

    static auto opCall()
    {
        NormalDistTest r;
        r.dist = NormalDist!(T, nlayers)(mean, sigma);
        return r;
    }

    T sample(Rng)(ref Rng rng)
    {
        return dist.get(rng) * 1;   
    }

    mixin NormalMixin!();
}

struct Bins(T)
{
    T low;
    T dx;
    T invDx;
    size_t n;
   
    this(T low, T high, size_t n)
    {
        this.low = low;
        this.n = n;
        dx = (high - low) / (n - 2);
        invDx = 1 / dx;
    }
 
    T lowerBound(size_t i)
    {
        assert(i != 0, "The first bin does not have a lower bound!");
        assert(i < n, "Bin index to high.");
        return low + (i - 1) * dx;
    }

    size_t index(T x)
    {
        return max(0, min(cast(ptrdiff_t)((x - low + dx) * invDx), n - 1));
    }
}

template histogram(DistTest, Rng)
{
    alias typeof(DistTest.init.sample(Rng.init)) T;
    
    auto histogram(ulong nsamples, Bins!T bins)
    {
        static auto zero(size_t nbins){ return new uint[nbins]; }

        static uint[] mapper(Tuple!(ulong, Bins!T) arg)
        {
            auto r = zero(arg[1].n);
            auto rng = Rng(unpredictableSeed);
            auto dist = DistTest();

            foreach(_; 0 .. arg[0])
                r[arg[1].index(dist.sample(rng))] ++;
            
            return r;
        }

        return mapper(tuple(nsamples, bins));
    }
}

void goodnessOfFit(T, DistTest, Rng)(ulong nsamples)
{
    auto nbins = to!size_t(nsamples ^^ (3.0 / 5.0));
    T expected = to!T(nsamples) / nbins;
    auto hist = new uint[nbins];
   
    static void worker(uint[] hist, size_t nbins, size_t nsamples)
    {  
        auto rng = Rng(unpredictableSeed);
        auto dt = DistTest();
        foreach(i; 0 .. nsamples)
            hist[to!size_t(dt.cdf(dt.sample(rng)) * nbins)]++;
    }

    import core.thread;

    enum nworkers = 4;
    auto histograms = 
        (nsamples / nworkers).repeat(nworkers).map!"new uint[a]".array;

    foreach(h; histograms)
    (h){
        (new Thread(() => worker(h, nbins, nsamples / nworkers))).start(); 
    }(h);

    thread_joinAll();

    foreach(h; histograms)
        hist[] += h[]; 

    stderr.writeln(nbins); 
    writeln(chiSquareFit(hist, repeat(1.0)));
}

void speed(T, DistTest, Rng)(ulong nsamples)
{
    auto rng = Rng(unpredictableSeed);

    auto dist = DistTest();

    T sum = 0;
    foreach(i; 0 .. nsamples)
    {
        auto x = dist.sample(rng);
        sum += x; 
    }
    writeln(sum);
} 

void main(string[] args)
{
    bool testSpeed;
    bool testError;
    getopt(args, "s", &testSpeed, "e", &testError);
   
    auto nsamples = args.length > 1 ? 
        to!ulong(args[1]) * 1000 : 1000_000_000L;
   
    alias double T; 
    alias NormalDistTest!(T, 128) DistTest;
    //alias DstatsNormalTest!T DistTest;
    //alias Xorshift128 Rng;
    alias Mt19937 Rng;

    if(testSpeed)
        speed!(T, DistTest, Rng)(nsamples);
    else
        goodnessOfFit!(T, DistTest, Rng)(nsamples);
}

