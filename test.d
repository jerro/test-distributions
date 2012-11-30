import std.stdio, std.mathspecial, std.algorithm, std.range, std.conv, 
    std.traits, std.numeric, std.random, std.typecons, std.typetuple,
    std.parallelism, std.getopt;

import core.bitop;

auto chiSquareP(R1, R2)(R1 sampled, R2 expected)
{
    auto scale = 
        sampled.reduce!"a+b".to!double / expected.reduce!"a+b".to!double;

    auto chisq = zip(sampled, expected, scale.repeat)
        .map!(a => (a[0] - a[1] * a[2]) ^^ 2 / (a[1] * a[2]))
        .reduce!"a+b";

    return gammaIncompleteCompl((sampled.length - 1) / 2, chisq / 2);
}

real normalCDF(real x, real mean, real sigma)
{
    return (1 + erf((x - mean) / (SQRT2 * sigma))) / 2;
}

mixin template NormalMixin()
{
    real cdf(real x)
    {
        return normalCDF(x, mean, sigma); 
    } 
}

struct Uniform(T)
{
    enum T width = 0.9;
    enum T invWidth = 1 / width;

    T sample(Rng)(ref Rng rng) @system
    {
        return uniform(0, width, rng);
        // return fastUniformFloat!T(rng) * width;
        // return accurateUniformFloat(width.to!T, rng);
    }

    real cdf(real x){ return x * invWidth; }
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

auto autoFindRoot(F)(F f)
{
    alias ReturnType!F T;
    T a = 0;
    T b = 1;

    while(f(a) * f(b) > 0)
    {
        auto dif = b - a;
        b += dif;
        a -= dif;
    }

    return findRoot(f, a, b);
}

auto binProbabilities(T, Cdf)(Cdf cdf, size_t nbins)
{
    double[] r;
    double prevCdf = 0;
    foreach(i; 1 .. nbins)
    // wrapping the body in a delegate avoids a segfault in f below
    (i){
        auto f = (real a) => cdf(a) - i.to!real / nbins;
        T x = autoFindRoot(f);
        if(f(x) < 0)
            x = nextUp(x);
       
        double currCdf = cdf(x); 
        r ~= currCdf - prevCdf;
        prevCdf = currCdf;
    }(i);
    
    r ~= 1 - prevCdf;
    return r;
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

    import core.thread, core.cpuid;

    auto nworkers = coresPerCPU;
    auto histograms = nbins.repeat(nworkers).map!"new uint[a]".array;

    foreach(h; histograms)
    (h){
        (new Thread(() => worker(h, nbins, nsamples / nworkers))).start(); 
    }(h);

    thread_joinAll();

    foreach(h; histograms)
        hist[] += h[];

    return chiSquareP(hist, binProbabilities!T(&(DistTest()).cdf, nbins));
}

auto goodnessOfFit(T, DistTest, Rng)(ulong samples)
{
    double sum = 0;
    double n = 0;
    auto avg = () => sum / n;
    do
    {
        sum += log10(goodnessOfFitImpl!(T, DistTest, Rng)(samples));
        n += 1;
        stderr.writefln("current average of log10(p): %s", avg());
    }
    while(avg() > -6 && avg() < -1 && n < 5);

    return avg() >= -1;
}

void speed(T, DistTest, Rng)(ulong nsamples)
{
    import std.datetime : StopWatch;

    auto rng = Rng(unpredictableSeed);

    auto dist = DistTest();

    StopWatch sw;
    sw.start();
    T sum = 0;
    foreach(i; 0 .. nsamples)
    {
        auto x = dist.sample(rng);
        sum += x;
    }
    sw.stop(); 
    writeln(sum);
    writefln("%s M/s", nsamples.to!double / sw.peek().usecs());
} 

void main(string[] args)
{
    bool testSpeed;
    bool testError;
    getopt(args, "s", &testSpeed, "e", &testError);
   
    auto nsamples = args.length > 1 ?  2 ^^ to!ulong(args[1]) : 0;
   
    alias double T; 
    //alias NormalZigguratEngine128 Engine;
    alias NormalBoxMullerEngine Engine;
   
    //alias Uniform!T DistTest; 
    alias NormalTest!(T, Engine, false) DistTest;
    
    //alias Xorshift128 Rng;
    alias Mt19937 Rng;

    if(testSpeed)
        speed!(T, DistTest, Rng)(nsamples);
    else
    {
        if(nsamples)
            writeln(goodnessOfFit!(T, DistTest, Rng)(nsamples) ? 
                "succeeded" : "failed");
        else
        {
            int i = 10;
            while(goodnessOfFit!(T, DistTest, Rng)(2UL ^^ i))
            {
                stderr.writefln("The test for nsamples == 2 ^^ %s succeeded", i);
                i++;
            }

            writefln("Failed at nsamples == 2 ^^ %s", i);
        }
    }
}
