use image::{GrayImage, ImageBuffer, RgbImage};
use ndarray::{azip, Array, Array2, Array3};
use num::Complex;
use num::Zero;
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

fn main() {
    match try_main() {
        Ok(()) => (),
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(1);
        }
    }
}

#[derive(Copy, Clone, Deserialize, Serialize)]
pub struct ComplexArea {
    min_corner: Complex<f64>,
    max_corner: Complex<f64>,
    res_re: usize,
    res_im: usize,
}

impl ComplexArea {
    fn discretize(&self, c: Complex<f64>) -> Option<[usize; 2]> {
        let normal_re = (c.re - self.min_corner.re) / (self.max_corner.re - self.min_corner.re);
        let normal_im = (c.im - self.min_corner.im) / (self.max_corner.im - self.min_corner.im);
        let re = normal_re * self.res_re as f64;
        let im = normal_im * self.res_im as f64;
        if re < 0.0 || im < 0.0 || re >= self.res_re as f64 || im >= self.res_im as f64 {
            return None;
        }
        Some([re as usize, im as usize])
    }
}

struct DenseMandelbrotComputation {
    area: ComplexArea,
    c: Array2<Complex<f64>>,
    z: Array2<Complex<f64>>,
    cur_iter: usize,
}

impl DenseMandelbrotComputation {
    fn new(area: ComplexArea) -> Self {
        let space_re = Array::linspace(area.min_corner.re, area.max_corner.re, area.res_re);
        let space_im = Array::linspace(area.min_corner.im, area.max_corner.im, area.res_im);
        let space_re = space_re
            .into_shape((area.res_re, 1))
            .unwrap()
            .broadcast((area.res_re, area.res_im))
            .unwrap()
            .map(|&re| Complex::new(re, 0.0));
        let space_im = space_im
            .broadcast((area.res_re, area.res_im))
            .unwrap()
            .map(|&im| Complex::new(0.0, im));
        let c = space_re + space_im;
        let z = Array2::zeros((area.res_re, area.res_im));
        Self {
            area,
            c,
            z,
            cur_iter: 0,
        }
    }

    fn run(&mut self, steps: usize) {
        for _i in 0..steps {
            self.cur_iter += 1;
            azip!((z in &mut self.z, c in &self.c) *z = (*z)*(*z) + c);
        }
    }

    fn finish(&self) -> MandelbrotSet {
        MandelbrotSet {
            area: self.area,
            diverged: self.z.map(|&z| z.is_nan() || z.norm_sqr() > 4.0),
            c: self.c.clone(),
        }
    }
}

struct Orbit {
    c: Complex<f64>,
    z: Complex<f64>,
}

impl Orbit {
    pub fn new(c: Complex<f64>) -> Self {
        Self {
            c,
            z: Complex::zero(),
        }
    }

    pub fn tick(&mut self) {
        self.z = self.z * self.z + self.c
    }

    pub fn diverged(&self) -> bool {
        self.z.norm_sqr() > 4.0
    }
}

type Extra = f64;

struct OrbitExtra {
    orbit: Orbit,
    extra: Extra,
}

use std::ops::{Deref, DerefMut};
impl Deref for OrbitExtra {
    type Target = Orbit;
    fn deref(&self) -> &Self::Target {
        &self.orbit
    }
}

impl DerefMut for OrbitExtra {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.orbit
    }
}

#[derive(Serialize, Deserialize)]
struct MandelbrotSet {
    area: ComplexArea,
    diverged: Array2<bool>,
    c: Array2<Complex<f64>>,
}

impl MandelbrotSet {
    fn make_iterations(&self) -> Vec<OrbitExtra> {
        let mut orbits = Vec::with_capacity(self.area.res_re * self.area.res_im / 4);
        azip!((&c in &self.c, &diverged in &self.diverged) {
            if diverged {
                let orbit = OrbitExtra { orbit: Orbit::new(c), extra: c.re };
                orbits.push(orbit);
            }
        });
        orbits
    }

    fn save_dbg_image(&self) {
        let buf = self.diverged.map(|&d| if d { 255u8 } else { 0u8 });
        let img: GrayImage = ImageBuffer::from_raw(
            u32::try_from(self.area.res_im).unwrap(),
            u32::try_from(self.area.res_re).unwrap(),
            buf.into_raw_vec(),
        )
        .unwrap();
        img.save("debug.png").unwrap();
    }
}

struct DensityMap {
    orbits: Vec<OrbitExtra>,
    area: ComplexArea,
    density: Array2<u64>,
    re: Array2<f64>,
}

impl DensityMap {
    pub fn new(area: ComplexArea, orbits: Vec<OrbitExtra>) -> Self {
        Self {
            orbits,
            area,
            density: Array2::zeros((area.res_re, area.res_im)),
            re: Array2::zeros((area.res_re, area.res_im)),
        }
    }

    fn register(&mut self, z: Complex<f64>, extra: Extra) {
        match self.area.discretize(z) {
            Some(coords) => {
                self.density[coords] += 1;
                self.re[coords] += extra;
            }
            None => (),
        }
    }

    pub fn run(mut self) -> Array3<u8> {
        let orbits = std::mem::replace(&mut self.orbits, Vec::new());
        for mut orb in orbits {
            while !orb.diverged() {
                orb.tick();
                self.register(orb.z, orb.extra);
            }
        }

        let max_density = self.density.fold(0, |a, b| a.max(*b));
        let density = self.density.mapv(|d| d as f64 / max_density as f64);
        let max_colorness = self.re.fold(0.0, |a: f64, b| a.max(*b));
        let colorness = self.re.map(|d| d / max_colorness);
        // Curve
        let density = density.mapv(|x| x.powf(0.33));
        let mut buf = Array3::zeros((density.shape()[0], density.shape()[1], 3));
        azip!((mut pix in buf.genrows_mut(), &r in &density, &g in &colorness) {
            pix[0] = (255.0 * r) as u8;
            // pix[1] = (255.0 * g) as u8;
            // pix[2] = (255.0 * b) as u8;
        });

        buf
    }
}

const MANDELBROT_FILENAME: &str = "mandelbrot.bc";

fn try_main() -> Result<(), String> {
    // Step 1: Calculate Mandelbrot set – this may have differing density from the final image.
    let mandelbrot_path = Path::new(MANDELBROT_FILENAME);
    if !mandelbrot_path.exists() {
        let num_iterations = 5000;
        let area = ComplexArea {
            min_corner: Complex::new(-2.0, -1.25),
            max_corner: Complex::new(1.0, 1.25),
            res_re: 12000,
            res_im: 10000,
        };
        println!(
            "Calculating Mandelbrot set for resolution {} x {} with {} iterations",
            area.res_re, area.res_im, num_iterations
        );

        let mut mb = DenseMandelbrotComputation::new(area);
        mb.run(num_iterations);
        let diverged = mb.finish();
        diverged.save_dbg_image();

        println!("Done");

        let file =
            BufWriter::new(File::create(mandelbrot_path).map_err(|_| "Could not create file.")?);
        bincode::serialize_into(file, &diverged).map_err(|_| "Could not serialize Mandelbrot.")?;
    }

    let file = BufReader::new(File::open(mandelbrot_path).map_err(|_| "Could not open file.")?);
    let diverged: MandelbrotSet =
        bincode::deserialize_from(file).map_err(|_| "Could not deserialize Mandelbrot.")?;

    // Step 2: Calculate orbits for points outside the mandelbrot set
    let orbits = diverged.make_iterations();
    println!("Iterating {} diverging orbits.", orbits.len());
    let render_area = ComplexArea {
        res_re: 3000,
        res_im: 2500,
        ..diverged.area
    };
    let dm = DensityMap::new(render_area, orbits);
    let buf = dm.run();
    // Step 3: Color mapping (todo: interactive?)
    // Step 4: Save image
    let img: RgbImage = ImageBuffer::from_raw(
        u32::try_from(render_area.res_im).unwrap(),
        u32::try_from(render_area.res_re).unwrap(),
        buf.into_raw_vec(),
    )
    .unwrap();
    img.save("buddhabrot.png").unwrap();
    Ok(())
}
