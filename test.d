import std.stdio, std.mathspecial, std.algorithm, std.range, std.conv, 
    std.traits, std.numeric, std.random, std.typecons, std.typetuple,
    std.parallelism, std.getopt;

import dstats.all;

struct SimpleNormal(T)
{
    T mean = 0;
    T sigma = 1;
    T cached;
    bool haveCached;

    auto sample(Rng)(ref Rng rng) @system
    {
        if(haveCached)
        {
            haveCached = false;
            return cached;
        }
        else
        {
            haveCached = true;
            while(true)
            {
                auto u = randomFloat!T(rng);
                auto v = randomFloat!T(rng);
                auto s = u * u + v * v;

                if(s > 1)
                    continue;

                import core.stdc.math;
                auto k = sqrt(- 2 * core.stdc.math.log(s) / s) * sigma;

                cached = k * u + mean;
                return   k * v + mean;
            }
        }
    }

    auto cdf(T x)
    {
        return normalCDF(x, mean, sigma); 
    } 
}
    
auto simpleNormal(T)(T mean, T sigma)
{
    return SimpleNormal!T(mean, sigma, T.init, false);
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
        return dist.get(rng);   
    }

    T cdf(T x)
    {
        return normalCDF(x, mean, sigma); 
    }

    T pdfMax()
    {
        return 1 / (sigma * sqrt(2 * PI));
    } 
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
        return max(0, min(cast(size_t)((x - low + dx) * invDx), n - 1));
    }
}

template sampleDistribution(DistTest, Rng)
{
    alias typeof(DistTest.init.sample(Rng.init)) T;
    
    auto sampleDistribution(ulong nsamples, Bins!T bins)
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

        /*enum  nchunks = 4;
        ulong samplesPerChunk = nsamples / nchunks;

        static auto reducer(const(uint)[] a, const(uint)[] b)
        {
            auto r = b.dup;
            r[] += a[];
            return r;
        }

        auto args = tuple(samplesPerChunk, bins).repeat(nchunks).array;
        args[0][0] += nsamples - nchunks * samplesPerChunk; 
        auto chunks = taskPool.amap!mapper(args);
        auto r = reduce!reducer(zero(bins.n), chunks);
        return r;*/

        return mapper(tuple(nsamples, bins));
    }
}

auto autoFindRoot(T)(scope T delegate(T) f)
{
    T a = 1;
    while(f(a) < 0)
        a += a;

    T b = -1;
    while(f(b) > 0)
        b += b;

    return findRoot(f, a, b); 
}

void goodnessOfFit(T, DistTest, Rng)(ulong nsamples)
{
    auto dt = DistTest();
    auto nEqualBins = to!size_t(nsamples ^^ (3.0 / 5.0));
    T equalBinArea = to!T(1) / nEqualBins;
    T binWidth = equalBinArea / dt.pdfMax();
    T high = autoFindRoot(delegate (T x) => dt.cdf(x) - (1 - equalBinArea));
    
    auto bins = Bins!T(-high, high, to!size_t(ceil(2 * high / binWidth)));

    auto dist = sampleDistribution!(DistTest, Rng)(nsamples, bins);

    auto getCdf = (size_t i) => 
        i == 0 ? 0 : 
        i == bins.n ? 1 : dt.cdf(bins.lowerBound(i));

    T chiSq = 0;
    size_t nbins = 0;
    for(size_t i; i < bins.n;)
    {
        T area = 0;
        uint n = 0;
        for(; area < equalBinArea && i < bins.n; i++)
        {
            area += getCdf(i + 1) - getCdf(i);
            n += dist[i];
        }
        
        T expected = area * nsamples;
        chiSq += (n - expected) ^^ 2 / expected; 
        nbins++;
    }
    
    stderr.writeln(chiSq);
    stderr.writeln(chiSquareCDFR(chiSq, nbins - 1));
}

void error(T, DistTest, Rng)(ulong nsamples)
{
    auto dist = sampleDistribution!(DistTest, Rng)(nsamples);
   
    auto maxError =  double.min;
    auto dt = DistTest();
    foreach(i, _; dist)
    {
        auto expected = 
            dt.cdf((i + 1) * maxX / nBuckets) - dt.cdf(i * maxX / nBuckets);
       
        auto nExpected = expected * nsamples;
        if(nExpected < 100)
            continue;

        auto sigma = sqrt(nExpected) / nsamples;
        maxError = max(maxError, abs(expected - dist[i]) / sigma);
    }

    writeln(maxError);
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

void plotLayers()()
{
    alias double T;
    mixin Normal!T;
    alias ZigguratTable!(f, fint, fderiv, 0.5, 32) Z;

    auto x = iota(1000).map!(a => 0.01 * a).array;
    auto y = x.map!f().array;
    
    auto layerx = (size_t i) =>
        i == 0 ? 0 :
        i == Z.nlayers ? Z.tailX : 
        i == Z.nlayers + 1 ? T.max : Z.layers[Z.nlayers - i].x;

    auto layery = (size_t i) => i == Z.nlayers + 1 ? 0 : f(layerx(i));
    
    auto ymin = new T[x.length];
    auto ymax = ymin.dup;

    int layer = 0;
    foreach(i, _; x)
    {
        if(x[i] > layerx(layer + 1))
            layer++;
        
        ymin[i] = layery(layer + 1);
        ymax[i] = layery(layer);
        
        writefln("%s\t%s\t%s\t%s", x[i], ymin[i], ymax[i], y[i]);
    }

    import plot2kill.all;
    
    Figure()
        .addPlot(LineGraph(x, y))
        .addPlot(LineGraph(x, ymin))
        .addPlot(LineGraph(x, ymax))
        .showAsMain();
}

void main(string[] args)
{
    bool testSpeed;
    bool testError;
    getopt(args, "s", &testSpeed, "e", &testError);
   
    auto nsamples = args.length > 1 ? 
        to!ulong(args[1]) * 1000 : 1000_000_000L;
   
    alias float T; 
    alias NormalDistTest!(T, 128) DistTest;
    alias Xorshift128 Rng;
    //alias Mt19937 Rng;

    if(testSpeed)
        speed!(T, DistTest, Rng)(nsamples);
    //else if(testError)
    //    error!(float, DistTest, Rng)(nsamples);
    else
        distribution!(T, DistTest, Rng)(nsamples);

    //plotLayers();
}

