const lightbox = document.createElement('div');
lightbox.className = 'lightbox';
lightbox.innerHTML = '<button aria-label="Close image">×</button><img alt="Expanded figure" />';
document.body.appendChild(lightbox);
const lightboxImg = lightbox.querySelector('img');
const closeBtn = lightbox.querySelector('button');

document.querySelectorAll('.gallery-item img, .image-card img, .hero-media img').forEach(img => {
  img.addEventListener('click', () => {
    lightboxImg.src = img.src;
    lightboxImg.alt = img.alt;
    lightbox.classList.add('open');
  });
});

function closeLightbox() {
  lightbox.classList.remove('open');
  lightboxImg.src = '';
}
closeBtn.addEventListener('click', closeLightbox);
lightbox.addEventListener('click', event => {
  if (event.target === lightbox) closeLightbox();
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape') closeLightbox();
});
