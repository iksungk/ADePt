# Accelerated Deep self-supervised Ptycho-laminography (ADePt)
This is the training code for the self-supervised deep learning model called Accelerated Deep self-supervised Ptycho-laminography (ADePt) for three-dimensional nanoscale X-ray imaging, specifically for X-ray ptycho-laminography. We aimed for minimizing data acquisition and computation time using self-supervised machine learning. The paper is originally published in <a href="https://doi.org/10.1364/OPTICA.492666">Optica</a>, and the arXiv paper is available <a href="https://arxiv.org/abs/2304.04597">here</a>.

## **Abstract**
Three-dimensional inspection of nanostructures such as integrated circuits is important for security and reliability assurance. Two scanning operations are required: ptychographic to recover the complex transmissivity of the specimen; and rotation of the specimen to acquire multiple projections covering the 3D spatial frequency domain. Two types of rotational scanning are possible: tomographic and laminographic. For flat, extended samples, for which the full 180 degree coverage is not possible, the latter is preferable because it provides better coverage of the 3D spatial frequency domain compared to limited-angle tomography. It is also because the amount of attenuation through the sample is approximately the same for all projections. However, both techniques are time consuming because of extensive acquisition and computation time. Here, we demonstrate the acceleration of ptycho-laminographic reconstruction of integrated circuits with 16-times fewer angular samples and 4.67-times faster computation by using a physics-regularized deep self-supervised learning architecture. We check the fidelity of our reconstruction against a densely sampled reconstruction that uses full scanning and no learning. As already reported elsewhere [Zhou and Horstmeyer, Opt. Express, 28(9), pp. 12872-12896], we observe improvement of reconstruction quality even over the densely sampled reconstruction, due to the ability of the self-supervised learning kernel to fill the missing cone.

## Citation
If you find the paper useful in your research, please consider citing the paper:


	@misc{kang2023accelerated,
      title={Accelerated deep self-supervised ptycho-laminography for three-dimensional nanoscale imaging of integrated circuits}, 
      author={Iksung Kang and Yi Jiang and Mirko Holler and Manuel Guizar-Sicairos and A. F. J. Levi and Jeffrey Klug and Stefan Vogt and George Barbastathis},
      year={2023},
      eprint={2304.04597},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}

## Data availability
The data that support the findings of this study are available from IARPA but restrictions apply to the availability of these data, which were used under license for the current study, and so are not publicly available. Data are however available from the authors upon reasonable request and with permission of IARPA.
