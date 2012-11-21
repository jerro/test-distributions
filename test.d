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
    enum T mean = 0;
    enum T sigma = 1;

    T sample(Rng)(ref Rng rng) @system
    {
        return rNorm(mean, sigma, rng);
    }
    
    mixin NormalMixin!();
}

struct NormalTest(T, alias Engine, bool useGlobal)
{
    enum T mean = 0;
    enum T sigma = 1;

    static if(useGlobal)
    {
        T sample(Rng)(ref Rng rng)
        {
            return normal!Engine(mean, sigma, rng);   
        }
    }
    else
    {
        Normal!(T, Engine) dist;

        static auto opCall()
        {
            NormalTest r;
            r.dist = normalRNG!Engine(mean, sigma);
            return r;
        }

        T sample(Rng)(ref Rng rng)
        {
            return dist.opCall(rng);   
        }
    }

    mixin NormalMixin!();
}

double goodnessOfFitImpl(T, DistTest, Rng)(ulong nsamples)
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
    auto histograms = nbins.repeat(nworkers).map!"new uint[a]".array;

    foreach(h; histograms)
    (h){
        (new Thread(() => worker(h, nbins, nsamples / nworkers))).start(); 
    }(h);

    thread_joinAll();

    foreach(h; histograms)
        hist[] += h[]; 

    return chiSquareFit(hist, repeat(1.0));
}

auto goodnessOfFit(T, DistTest, Rng)(ulong samples)
{
    double sum = 0;
    double n = 0;
    do
    {
        sum += log10(goodnessOfFitImpl!(T, DistTest, Rng)(samples));
        n += 1;
    }
    while(sum / n > -6 && sum / n < -1);

    return 10 ^^ (sum / n);
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
    alias NormalZigguratEngine64 Engine;
    //alias NormalBoxMullerEngine Engine;
    alias NormalTest!(T, Engine, true) DistTest;
    //alias Xorshift128 Rng;
    alias Mt19937 Rng;

    if(testSpeed)
        speed!(T, DistTest, Rng)(nsamples);
    else
        writeln(goodnessOfFit!(T, DistTest, Rng)(nsamples));
}

