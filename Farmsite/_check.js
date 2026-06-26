
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const LS={cat:'orchard_catalog_v1',cart:'orchard_cart_v1',set:'orchard_settings_v1'};

const DEFAULT_FRUITS=[
 {id:1,name:"Royal Gala Apples",cat:"Orchard",price:189,unit:"per kg",emoji:"🍎",stock:true,desc:"Crisp, sweet and snappy — straight from the hills.",img:"https://images.unsplash.com/photo-1568702846914-96b305d2aaeb?w=600&q=80&auto=format&fit=crop"},
 {id:2,name:"Alphonso Mangoes",cat:"Tropical",price:399,unit:"per kg",emoji:"🥭",stock:true,desc:"The king of fruits — rich, fragrant and buttery.",img:"https://images.unsplash.com/photo-1605027990121-cbae9e0642df?w=600&q=80&auto=format&fit=crop"},
 {id:3,name:"Nagpur Oranges",cat:"Citrus",price:129,unit:"per kg",emoji:"🍊",stock:true,desc:"Juicy, tangy-sweet and bursting with vitamin C.",img:"https://images.unsplash.com/photo-1547514701-42782101795e?w=600&q=80&auto=format&fit=crop"},
 {id:4,name:"Fresh Strawberries",cat:"Berries",price:249,unit:"per box",emoji:"🍓",stock:true,desc:"Plump, ruby-red and hand-picked at dawn.",img:"https://images.unsplash.com/photo-1518635017498-87f514b751ba?w=600&q=80&auto=format&fit=crop"},
 {id:5,name:"Seedless Grapes",cat:"Orchard",price:159,unit:"per kg",emoji:"🍇",stock:true,desc:"Cool, crunchy green grapes — a perfect snack.",img:"https://images.unsplash.com/photo-1537640538966-79f369143f8f?w=600&q=80&auto=format&fit=crop"},
 {id:6,name:"Sweet Pineapple",cat:"Tropical",price:99,unit:"each",emoji:"🍍",stock:true,desc:"Golden, tangy and tropical — sunshine in a fruit.",img:"https://images.unsplash.com/photo-1550258987-190a2d41a8ba?w=600&q=80&auto=format&fit=crop"},
 {id:7,name:"Baby Watermelon",cat:"Melons",price:79,unit:"each",emoji:"🍉",stock:true,desc:"Crisp, hydrating and impossibly refreshing.",img:"https://images.unsplash.com/photo-1587049352846-4a222e784d38?w=600&q=80&auto=format&fit=crop"},
 {id:8,name:"Wild Blueberries",cat:"Berries",price:329,unit:"per box",emoji:"🫐",stock:false,desc:"Antioxidant-rich little gems. Back in stock soon.",img:"https://images.unsplash.com/photo-1498557850523-fd3d118b962e?w=600&q=80&auto=format&fit=crop"},
 {id:9,name:"Ruby Pomegranate",cat:"Exotic",price:219,unit:"per kg",emoji:"🍎",stock:true,desc:"Jewel-like arils, sweet-tart and heart-healthy.",img:"https://images.unsplash.com/photo-1541344999736-83eca272f6fc?w=600&q=80&auto=format&fit=crop"},
 {id:10,name:"Golden Kiwi",cat:"Exotic",price:289,unit:"per kg",emoji:"🥝",stock:true,desc:"Tropical, tangy and loaded with vitamin C.",img:"https://images.unsplash.com/photo-1585059895524-72359e06133a?w=600&q=80&auto=format&fit=crop"},
 {id:11,name:"Cavendish Bananas",cat:"Tropical",price:69,unit:"per dozen",emoji:"🍌",stock:true,desc:"Naturally ripened, creamy and energy-packed.",img:"https://images.unsplash.com/photo-1571771894821-ce9b6c11b08e?w=600&q=80&auto=format&fit=crop"},
 {id:12,name:"Juicy Peaches",cat:"Stone",price:269,unit:"per kg",emoji:"🍑",stock:true,desc:"Soft, fragrant and dripping with summer sweetness.",img:"https://images.unsplash.com/photo-1629828874514-c1e5103f8b7f?w=600&q=80&auto=format&fit=crop"}
];

let fruits=load(LS.cat,DEFAULT_FRUITS);
let cart=load(LS.cart,{});
let settings=load(LS.set,{wa:"919876543210"});
let activeCat="All", query="", isAdmin=false;

function load(k,f){try{const v=JSON.parse(localStorage.getItem(k));return v??f}catch(e){return f}}
function save(k,v){localStorage.setItem(k,JSON.stringify(v))}
function money(n){return "₹"+Number(n).toLocaleString('en-IN')}
function toast(m){const t=$('#toast');t.textContent=m;t.classList.add('show');clearTimeout(t._t);t._t=setTimeout(()=>t.classList.remove('show'),2200)}

function categories(){return ["All",...new Set(fruits.map(f=>f.cat))]}
function renderChips(){
  $('#chips').innerHTML=categories().map(c=>`<button class="chip${c===activeCat?' active':''}" data-cat="${c}">${c}</button>`).join('');
}
function renderGrid(){
  const list=fruits.filter(f=>(activeCat==="All"||f.cat===activeCat)&&f.name.toLowerCase().includes(query));
  const g=$('#grid');
  if(!list.length){g.innerHTML=`<div class="empty"><div class="big">🍂</div>No fruits match your search.</div>`;return}
  g.innerHTML=list.map(f=>`
    <article class="card reveal in" data-id="${f.id}">
      <div class="card-img${f.img?'':' broken'}">
        ${f.img?`<img src="${f.img}" alt="${f.name}" loading="lazy" onerror="this.closest('.card-img').classList.add('broken')">`:''}
        <div class="fallback">${f.emoji||'🍎'}</div>
        <span class="tag${f.stock?'':' out'}">${f.stock?'🌿 Organic':'Sold out'}</span>
      </div>
      <div class="card-body">
        <span class="cat">${f.cat}</span>
        <h3>${f.name}</h3>
        <p class="desc">${f.desc||''}</p>
        <div class="price-row"><span class="price">${money(f.price)}</span><span class="unit">${f.unit||''}</span></div>
        <div class="card-actions">
          <button class="add-btn" data-add="${f.id}" ${f.stock?'':'disabled'}>${f.stock?'＋ Add to basket':'Out of stock'}</button>
          <button class="del-btn" data-del="${f.id}" title="Remove fruit">🗑️</button>
        </div>
      </div>
    </article>`).join('');
}
function renderAll(){renderChips();renderGrid();$('#statCount').textContent=fruits.length+'+';}

function cartArr(){return Object.values(cart)}
function cartQty(){return cartArr().reduce((s,i)=>s+i.qty,0)}
function cartTotal(){return cartArr().reduce((s,i)=>s+i.qty*i.price,0)}
function addToCart(id){
  const f=fruits.find(x=>x.id===id);if(!f||!f.stock)return;
  if(cart[id])cart[id].qty++;else cart[id]={id,name:f.name,price:f.price,unit:f.unit,emoji:f.emoji,img:f.img,qty:1};
  save(LS.cart,cart);renderCart();toast(`${f.name} added 🧺`);
}
function setQty(id,d){if(!cart[id])return;cart[id].qty+=d;if(cart[id].qty<=0)delete cart[id];save(LS.cart,cart);renderCart()}
function renderCart(){
  const c=$('#cartCount'),q=cartQty();
  c.textContent=q;c.classList.toggle('show',q>0);
  const items=cartArr(),box=$('#cartItems');
  $('#cartEmpty').style.display=items.length?'none':'grid';
  $('#cartFoot').style.display=items.length?'block':'none';
  box.innerHTML=items.map(i=>`
    <div class="ci">
      ${i.img?`<img src="${i.img}" alt="${i.name}" onerror="this.outerHTML='<div class=ph>${i.emoji||'🍎'}</div>'">`:`<div class="ph">${i.emoji||'🍎'}</div>`}
      <div class="ci-info"><b>${i.name}</b><span class="pu">${money(i.price)} ${i.unit||''}</span>
        <div class="qty"><button data-q="${i.id}" data-d="-1">−</button><span>${i.qty}</span><button data-q="${i.id}" data-d="1">＋</button></div>
      </div>
      <div class="ci-sub">${money(i.price*i.qty)}</div>
    </div>`).join('');
  $('#cartTotal').textContent=money(cartTotal());
}
function whatsappOrder(){
  const items=cartArr();if(!items.length)return;
  let msg=`Hi ${$('#brandName').textContent}! 🍎 I'd like to order:%0A%0A`;
  items.forEach(i=>{msg+=`• ${i.name} x${i.qty} — ${money(i.price*i.qty)}%0A`});
  msg+=`%0A*Total: ${money(cartTotal())}*%0A%0APlease confirm availability & delivery. Thank you!`;
  window.open(`https://wa.me/${settings.wa}?text=${msg}`,'_blank');
}

function openDrawer(el){$('#overlay').classList.add('show');el.classList.add('show')}
function closeDrawers(){$('#overlay').classList.remove('show');$('#cartDrawer').classList.remove('show');$('#adminDrawer').classList.remove('show')}

function enableAdmin(){
  isAdmin=true;document.body.classList.add('admin');
  $('#adminToggle').style.display='inline-flex';$('#setWa').value=settings.wa;
  toast('Owner mode on 🔧');
}
$('#adminLink').onclick=e=>{e.preventDefault();
  if(isAdmin){openDrawer($('#adminDrawer'));return}
  const p=prompt('Owner password:');
  if(p==='orchard2026'){enableAdmin();openDrawer($('#adminDrawer'))}
  else if(p!==null)toast('Wrong password');
};
$('#adminToggle').onclick=()=>openDrawer($('#adminDrawer'));
$('#addForm').onsubmit=e=>{
  e.preventDefault();
  const f={id:Date.now(),name:$('#fName').value.trim(),price:+$('#fPrice').value,unit:$('#fUnit').value.trim()||'per kg',
    cat:$('#fCat').value,emoji:$('#fEmoji').value.trim()||'🍎',img:$('#fImg').value.trim(),desc:$('#fDesc').value.trim(),stock:$('#fStock').value==='yes'};
  fruits.unshift(f);save(LS.cat,fruits);renderAll();e.target.reset();$('#fUnit').value='per kg';$('#fEmoji').value='🍎';
  toast(`${f.name} added to catalog ✓`);
};
$('#saveSettings').onclick=()=>{settings.wa=$('#setWa').value.replace(/\D/g,'');save(LS.set,settings);toast('Settings saved ✓')};
$('#resetBtn').onclick=()=>{if(confirm('Reset catalog to the original defaults?')){fruits=JSON.parse(JSON.stringify(DEFAULT_FRUITS));save(LS.cat,fruits);renderAll();toast('Catalog reset ↺')}};

document.addEventListener('click',e=>{
  const add=e.target.closest('[data-add]'); if(add){addToCart(+add.dataset.add);return}
  const del=e.target.closest('[data-del]'); if(del&&isAdmin){const id=+del.dataset.del;const f=fruits.find(x=>x.id===id);if(confirm(`Remove "${f.name}" from the catalog?`)){fruits=fruits.filter(x=>x.id!==id);save(LS.cat,fruits);renderAll();toast('Fruit removed')}return}
  const q=e.target.closest('[data-q]'); if(q){setQty(+q.dataset.q,+q.dataset.d);return}
  const chip=e.target.closest('.chip'); if(chip){activeCat=chip.dataset.cat;renderChips();renderGrid();return}
});
$('#search').oninput=e=>{query=e.target.value.toLowerCase().trim();renderGrid()};
$('#cartBtn').onclick=()=>openDrawer($('#cartDrawer'));
$('#heroCartBtn').onclick=()=>openDrawer($('#cartDrawer'));
$('#cartClose').onclick=closeDrawers;$('#adminClose').onclick=closeDrawers;$('#overlay').onclick=closeDrawers;
$('#waBtn').onclick=whatsappOrder;
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeDrawers()});
document.addEventListener('keydown',e=>{if(e.shiftKey&&e.key.toLowerCase()==='a'&&!/INPUT|TEXTAREA/.test(document.activeElement.tagName)){if(!isAdmin)enableAdmin();openDrawer($('#adminDrawer'))}});

const io=new IntersectionObserver(es=>es.forEach(x=>{if(x.isIntersecting){x.target.classList.add('in');io.unobserve(x.target)}}),{threshold:.12});
$$('.reveal').forEach(el=>io.observe(el));

renderAll();renderCart();
