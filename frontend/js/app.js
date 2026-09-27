/**
 * AI Research Collaborator — Frontend Application Logic
 * Modern, interactive client for vector matchmaking & paper recommendations.
 */

const API_BASE = "";

// Distinct pastel avatar background palettes
const AVATAR_COLORS = [
  "linear-gradient(135deg, #4f46e5 0%, #6366f1 100%)",
  "linear-gradient(135deg, #0d9488 0%, #14b8a6 100%)",
  "linear-gradient(135deg, #7c3aed 0%, #a855f7 100%)",
  "linear-gradient(135deg, #2563eb 0%, #38bdf8 100%)",
  "linear-gradient(135deg, #db2777 0%, #f472b6 100%)",
  "linear-gradient(135deg, #d97706 0%, #fbbf24 100%)",
  "linear-gradient(135deg, #059669 0%, #34d399 100%)"
];

function getAvatarGradient(name) {
  if (!name) return AVATAR_COLORS[0];
  let hash = 0;
  for (let i = 0; i < name.length; i++) {
    hash = name.charCodeAt(i) + ((hash << 5) - hash);
  }
  return AVATAR_COLORS[Math.abs(hash) % AVATAR_COLORS.length];
}

function getInitials(name) {
  if (!name) return "AI";
  const parts = name.trim().split(/\s+/);
  if (parts.length === 1) return parts[0].substring(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}

function getCategoryClass(cat) {
  const c = cat.toLowerCase();
  if (c.includes("cs.ai")) return "cat-cs-ai";
  if (c.includes("cs.lg")) return "cat-cs-lg";
  if (c.includes("cs.cv")) return "cat-cs-cv";
  if (c.includes("cs.cl")) return "cat-cs-cl";
  if (c.includes("cs.hc")) return "cat-cs-hc";
  if (c.includes("cs.ro")) return "cat-cs-ro";
  if (c.includes("cs.ne")) return "cat-cs-ne";
  return "";
}

class App {
  constructor() {
    this.currentTab = "matchmaker";
    
    // Pagination & filter state
    this.researcherPage = 1;
    this.researcherLimit = 12;
    this.researcherQuery = "";
    this.researcherCategory = "";

    this.paperPage = 1;
    this.paperLimit = 12;
    this.paperQuery = "";
    this.paperCategory = "";

    // Recommender state
    this.allResearchers = [];

    // Saved bookmarks
    this.savedResearchers = JSON.parse(localStorage.getItem("saved_researchers") || "[]");
    this.savedPapers = JSON.parse(localStorage.getItem("saved_papers") || "[]");

    // Cache for modal data
    this.paperCache = new Map();
    this.researcherCache = new Map();

    this.debounceTimer = null;
  }

  async init() {
    this.updateSavedBadge();
    await this.fetchStats();
    await this.loadResearchers();
    await this.populateResearcherDropdown();
    await this.loadPapers();
  }

  // Toast Notification
  showToast(msg, icon = "✨") {
    const container = document.getElementById("toast-container");
    const toast = document.createElement("div");
    toast.className = "toast";
    toast.innerHTML = `<span>${icon}</span><span>${msg}</span>`;
    container.appendChild(toast);
    setTimeout(() => {
      toast.style.transition = "opacity 0.3s ease, transform 0.3s ease";
      toast.style.opacity = "0";
      toast.style.transform = "translateX(50px)";
      setTimeout(() => toast.remove(), 300);
    }, 2800);
  }

  // Switch Active Tab
  switchTab(tabId) {
    this.currentTab = tabId;
    document.querySelectorAll(".tab-btn").forEach(btn => {
      btn.classList.toggle("active", btn.id === `tab-btn-${tabId}`);
    });
    document.querySelectorAll(".tab-view").forEach(view => {
      view.classList.toggle("active", view.id === `view-${tabId}`);
    });

    if (tabId === "saved") {
      this.renderSaved();
    }
  }

  // Fetch Dataset Statistics
  async fetchStats() {
    try {
      const res = await fetch(`${API_BASE}/api/stats`);
      if (!res.ok) return;
      const data = await res.json();
      document.getElementById("stat-papers").innerHTML = `<strong>${data.total_papers.toLocaleString()}</strong> Papers`;
      document.getElementById("stat-researchers").innerHTML = `${data.total_researchers} Researchers`;
      document.getElementById("stat-dims").innerHTML = `${data.embedding_dimensions}-dim Vectors`;
    } catch (err) {
      console.error("Failed to load stats", err);
    }
  }

  // Load Researcher Directory
  async loadResearchers() {
    const grid = document.getElementById("researchers-grid");
    grid.innerHTML = `<div class="skeleton-card"></div><div class="skeleton-card"></div><div class="skeleton-card"></div>`;

    try {
      const params = new URLSearchParams({
        page: this.researcherPage,
        limit: this.researcherLimit,
        q: this.researcherQuery,
        category: this.researcherCategory
      });
      const res = await fetch(`${API_BASE}/api/researchers?${params}`);
      const data = await res.json();

      document.getElementById("researcher-count").textContent = `Showing ${data.researchers.length} of ${data.total} researchers`;
      this.renderResearchers(data.researchers);
      this.renderResearchersPagination(data.page, data.total_pages);
    } catch (err) {
      console.error(err);
      grid.innerHTML = `<p style="color: #ef4444; padding: 2rem;">Failed to load researchers. Make sure server is running.</p>`;
    }
  }

  renderResearchers(researchers) {
    const grid = document.getElementById("researchers-grid");
    if (!researchers.length) {
      grid.innerHTML = `
        <div style="grid-column: 1 / -1; background: #ffffff; border: 1px dashed var(--border-color); border-radius: var(--radius-lg); padding: 3rem; text-align: center; color: var(--text-subtle);">
          <p style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">No researchers found</p>
          <p style="font-size: 0.9rem;">Try adjusting your keyword search or category filter.</p>
        </div>`;
      return;
    }

    grid.innerHTML = researchers.map(r => {
      this.researcherCache.set(r.id, r);
      const isSaved = this.savedResearchers.some(x => x.id === r.id);
      const categoriesHtml = (r.category_list || []).slice(0, 3).map(cat => 
        `<span class="badge-category ${getCategoryClass(cat)}">${cat}</span>`
      ).join(" ");

      return `
        <div class="researcher-card">
          <div class="researcher-header">
            <div class="researcher-avatar" style="background: ${getAvatarGradient(r.researcher_name)}">
              ${getInitials(r.researcher_name)}
            </div>
            <div class="researcher-meta">
              <h4 class="researcher-name">${r.researcher_name}</h4>
              <div class="researcher-stats-row">
                <span>📄 ${r.paper_count || 1} Paper${r.paper_count > 1 ? 's' : ''}</span>
                <span>•</span>
                <span>ID: #${r.id}</span>
              </div>
            </div>
          </div>

          <div class="card-categories">${categoriesHtml}</div>

          <p class="research-snippet">${r.research_text || "No research abstract available."}</p>

          <div class="card-actions">
            <button class="btn-action btn-accent" onclick="app.matchCollaborators('${encodeURIComponent(r.researcher_name)}')">
              <span>👥 Match Partners</span>
            </button>
            <button class="btn-action" onclick="app.quickRecommendPapers('${encodeURIComponent(r.researcher_name)}')">
              <span>📑 Papers</span>
            </button>
            <button class="btn-icon-square ${isSaved ? 'bookmarked' : ''}" onclick="app.toggleSaveResearcher(${r.id})" title="${isSaved ? 'Remove from Saved' : 'Save Researcher'}">
              ★
            </button>
          </div>
        </div>
      `;
    }).join("");
  }

  renderResearchersPagination(page, totalPages) {
    const container = document.getElementById("researchers-pagination");
    if (totalPages <= 1) {
      container.innerHTML = "";
      return;
    }
    container.innerHTML = `
      <button class="page-btn" ${page <= 1 ? "disabled" : ""} onclick="app.changeResearcherPage(${page - 1})">← Prev</button>
      <span class="page-info">Page ${page} of ${totalPages}</span>
      <button class="page-btn" ${page >= totalPages ? "disabled" : ""} onclick="app.changeResearcherPage(${page + 1})">Next →</button>
    `;
  }

  changeResearcherPage(newPage) {
    this.researcherPage = newPage;
    this.loadResearchers();
  }

  onResearcherSearch() {
    clearTimeout(this.debounceTimer);
    this.debounceTimer = setTimeout(() => {
      this.researcherQuery = document.getElementById("researcher-search-input").value;
      this.researcherPage = 1;
      this.loadResearchers();
    }, 250);
  }

  onResearcherCategoryFilter() {
    this.researcherCategory = document.getElementById("researcher-category-filter").value;
    this.researcherPage = 1;
    this.loadResearchers();
  }

  filterResearcherChip(cat) {
    this.researcherCategory = cat;
    document.getElementById("researcher-category-filter").value = cat;
    document.querySelectorAll("#researcher-category-chips .category-chip").forEach(btn => {
      btn.classList.toggle("active", (cat === "" && btn.textContent.includes("All")) || btn.textContent.includes(cat));
    });
    this.researcherPage = 1;
    this.loadResearchers();
  }

  // Collaborator Matchmaker: Find Similar Researchers
  async matchCollaborators(encodedName) {
    const name = decodeURIComponent(encodedName);
    const container = document.getElementById("matchmaker-active-container");
    container.style.display = "block";
    container.innerHTML = `
      <div style="background: #ffffff; border: 1px solid var(--border-color); border-radius: var(--radius-xl); padding: 2rem; text-align: center;">
        <div class="stat-dot" style="margin-right: 0.5rem;"></div> Calculating cosine similarity across all researcher embeddings for <strong>${name}</strong>...
      </div>
    `;

    // Scroll to top of match view
    container.scrollIntoView({ behavior: 'smooth', block: 'start' });

    try {
      const res = await fetch(`${API_BASE}/api/researchers/${encodeURIComponent(name)}/similar?limit=6`);
      if (!res.ok) throw new Error("Researcher not found");
      const data = await res.json();
      this.renderMatchmakerView(data);
    } catch (err) {
      console.error(err);
      container.innerHTML = `
        <div style="background: #ffffff; border: 1px solid #fca5a5; border-radius: var(--radius-xl); padding: 1.5rem; text-align: center; color: #dc2626;">
          Could not find matches for ${name}.
        </div>
      `;
    }
  }

  renderMatchmakerView(data) {
    const container = document.getElementById("matchmaker-active-container");
    const target = data.target_researcher;
    const collabs = data.collaborators;

    const targetCategoriesHtml = (target.categories || []).map(cat => 
      `<span class="badge-category ${getCategoryClass(cat)}">${cat}</span>`
    ).join(" ");

    const collaboratorsHtml = collabs.map((c, i) => {
      const matchClass = c.similarity >= 0.7 ? "match-high" : (c.similarity >= 0.5 ? "match-medium" : "match-moderate");
      const isSaved = this.savedResearchers.some(x => x.id === c.id);
      const sharedTagsHtml = (c.shared_categories || []).map(cat => 
        `<span class="badge-category ${getCategoryClass(cat)}" style="background: #ecfdf5; border-color: #a7f3d0; color: #047857;">🤝 ${cat}</span>`
      ).join(" ");

      return `
        <div class="researcher-card" style="box-shadow: var(--shadow-md);">
          <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.75rem;">
            <div style="display: flex; gap: 0.85rem; align-items: center;">
              <div class="researcher-avatar" style="background: ${getAvatarGradient(c.researcher_name)}; width: 44px; height: 44px; font-size: 1rem;">
                ${getInitials(c.researcher_name)}
              </div>
              <div>
                <h4 class="researcher-name" style="font-size: 1.05rem;">${c.researcher_name}</h4>
                <div class="researcher-stats-row">
                  <span>${c.paper_count} Paper${c.paper_count > 1 ? 's' : ''}</span>
                  <span>•</span>
                  <span>Rank #${i + 1} Match</span>
                </div>
              </div>
            </div>
            <div class="match-pill ${matchClass}">
              <span>🔥</span> ${c.match_percent}% Match
            </div>
          </div>

          <div class="card-categories" style="margin-bottom: 0.5rem;">
            ${sharedTagsHtml || '<span style="font-size: 0.75rem; color: var(--text-subtle);">Semantic Profile Match</span>'}
          </div>

          <p class="research-snippet" style="-webkit-line-clamp: 2; margin-bottom: 1rem;">${c.research_snippet}</p>

          <div class="card-actions">
            <button class="btn-action btn-accent" onclick="app.matchCollaborators('${encodeURIComponent(c.researcher_name)}')">
              <span>View Network</span>
            </button>
            <button class="btn-action" onclick="app.quickRecommendPapers('${encodeURIComponent(c.researcher_name)}')">
              <span>Tailored Papers</span>
            </button>
            <button class="btn-icon-square ${isSaved ? 'bookmarked' : ''}" onclick="app.toggleSaveResearcher(${c.id})" title="Save">
              ★
            </button>
          </div>
        </div>
      `;
    }).join("");

    container.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
        <h3 class="section-title"><span>🎯 Semantic Collaborator Match Results</span></h3>
        <button class="btn-secondary" style="padding: 0.4rem 0.85rem; font-size: 0.8rem;" onclick="document.getElementById('matchmaker-active-container').style.display='none'">
          Close Match View ✕
        </button>
      </div>

      <div class="matchmaker-layout">
        <div class="target-researcher-pane">
          <span class="target-badge">Selected Target Researcher</span>
          <div class="target-profile-avatar" style="background: ${getAvatarGradient(target.name)}">
            ${getInitials(target.name)}
          </div>
          <h3 class="target-profile-name">${target.name}</h3>
          <div class="target-info-row">
            <span>📄 ${target.paper_count || 1} Paper${target.paper_count > 1 ? 's' : ''}</span>
            <span>•</span>
            <span>ID: #${target.id}</span>
          </div>
          <div class="card-categories" style="margin-bottom: 1.25rem;">
            ${targetCategoriesHtml}
          </div>
          <button class="btn-primary" style="width: 100%; justify-content: center;" onclick="app.quickRecommendPapers('${encodeURIComponent(target.name)}')">
            <span>View Tailored arXiv Papers</span> 📑
          </button>
        </div>

        <div>
          <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 1.25rem;">
            ${collaboratorsHtml}
          </div>
        </div>
      </div>
    `;
  }

  // Populate Dropdown for Paper Recommender
  async populateResearcherDropdown() {
    try {
      const res = await fetch(`${API_BASE}/api/researchers?limit=100`);
      const data = await res.json();
      this.allResearchers = data.researchers;
      const select = document.getElementById("recommender-select-researcher");
      select.innerHTML = `<option value="">-- Select a researcher to get tailored paper recommendations --</option>` +
        this.allResearchers.map(r => `<option value="${r.researcher_name}">${r.researcher_name} (${r.paper_count || 1} paper)</option>`).join("");
    } catch (e) {
      console.error(e);
    }
  }

  quickRecommendPapers(encodedName) {
    const name = decodeURIComponent(encodedName);
    this.switchTab("recommender");
    const select = document.getElementById("recommender-select-researcher");
    select.value = name;
    this.loadPaperRecommendations(name);
  }

  onRecommenderSelectChange() {
    const name = document.getElementById("recommender-select-researcher").value;
    if (name) {
      this.loadPaperRecommendations(name);
    }
  }

  async loadPaperRecommendations(optName) {
    const name = optName || document.getElementById("recommender-select-researcher").value;
    if (!name) {
      this.showToast("Please choose a researcher first", "ℹ️");
      return;
    }

    const cat = document.getElementById("recommender-category-filter").value;
    const grid = document.getElementById("recommender-papers-grid");
    const countSpan = document.getElementById("recommender-results-count");
    grid.innerHTML = `<div class="skeleton-card"></div><div class="skeleton-card"></div>`;
    countSpan.textContent = "Computing vector dot products across 10,000 papers...";

    try {
      const params = new URLSearchParams({ limit: 12 });
      if (cat) params.append("category", cat);

      const res = await fetch(`${API_BASE}/api/researchers/${encodeURIComponent(name)}/recommended-papers?${params}`);
      if (!res.ok) throw new Error("Recommendations request failed");
      const data = await res.json();

      countSpan.textContent = `Found ${data.recommendations.length} tailored papers for ${data.researcher.name}`;

      // Update banner
      const banner = document.getElementById("recommender-researcher-banner");
      banner.style.display = "block";
      banner.innerHTML = `
        <div style="background: #ffffff; border: 1px solid var(--border-color); border-radius: var(--radius-lg); padding: 1.25rem 1.5rem; display: flex; align-items: center; justify-content: space-between; box-shadow: var(--shadow-sm);">
          <div style="display: flex; align-items: center; gap: 1rem;">
            <div class="researcher-avatar" style="background: ${getAvatarGradient(data.researcher.name)}; width: 44px; height: 44px;">
              ${getInitials(data.researcher.name)}
            </div>
            <div>
              <h4 style="font-size: 1.15rem; font-weight: 800; color: var(--text-main);">${data.researcher.name}</h4>
              <p style="font-size: 0.82rem; color: var(--text-subtle);">Showing highest cosine similarity papers aligned with this researcher's dense profile vector.</p>
            </div>
          </div>
          <button class="btn-action btn-accent" onclick="app.matchCollaborators('${encodeURIComponent(data.researcher.name)}')">
            <span>👥 Find Co-Authors</span>
          </button>
        </div>
      `;

      this.renderPaperCards(data.recommendations, grid, true);
    } catch (err) {
      console.error(err);
      grid.innerHTML = `<p style="color: #ef4444; padding: 2rem;">Failed to load recommended papers.</p>`;
    }
  }

  // Load ArXiv Papers (Explorer Tab)
  async loadPapers() {
    const grid = document.getElementById("papers-grid");
    grid.innerHTML = `<div class="skeleton-card"></div><div class="skeleton-card"></div><div class="skeleton-card"></div>`;

    try {
      const params = new URLSearchParams({
        page: this.paperPage,
        limit: this.paperLimit,
        q: this.paperQuery,
        category: this.paperCategory
      });
      const res = await fetch(`${API_BASE}/api/papers?${params}`);
      const data = await res.json();

      document.getElementById("papers-count").textContent = `Showing ${data.papers.length} of ${data.total.toLocaleString()} papers`;
      this.renderPaperCards(data.papers, grid, false);
      this.renderPapersPagination(data.page, data.total_pages);
    } catch (err) {
      console.error(err);
      grid.innerHTML = `<p style="color: #ef4444; padding: 2rem;">Failed to load papers.</p>`;
    }
  }

  renderPaperCards(papers, container, showSimilarity = false) {
    if (!papers.length) {
      container.innerHTML = `
        <div style="grid-column: 1 / -1; background: #ffffff; border: 1px dashed var(--border-color); border-radius: var(--radius-lg); padding: 3rem; text-align: center; color: var(--text-subtle);">
          <p style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">No papers found</p>
          <p style="font-size: 0.9rem;">Try searching for another topic or category.</p>
        </div>`;
      return;
    }

    container.innerHTML = papers.map(p => {
      this.paperCache.set(p.id, p);
      const isSaved = this.savedPapers.some(x => x.id === p.id);
      const categoriesHtml = (p.category_list || []).slice(0, 3).map(cat => 
        `<span class="badge-category ${getCategoryClass(cat)}">${cat}</span>`
      ).join(" ");

      let similarityBarHtml = "";
      if (showSimilarity && p.match_percent !== undefined) {
        similarityBarHtml = `
          <div class="similarity-bar-wrapper">
            <div class="similarity-bar-label">
              <span>Semantic Similarity</span>
              <span style="color: var(--primary);">${p.match_percent}% Match</span>
            </div>
            <div class="similarity-bar-bg">
              <div class="similarity-bar-fill" style="width: ${p.match_percent}%;"></div>
            </div>
          </div>
        `;
      }

      return `
        <div class="paper-card">
          <div>
            <div class="paper-top-meta">
              <span class="paper-date">📅 ${p.published || "Recent"}</span>
              <div>${categoriesHtml}</div>
            </div>

            <a href="${p.paper_id || '#'}" target="_blank" rel="noopener noreferrer" class="paper-title">
              ${p.title}
            </a>

            ${similarityBarHtml}

            <p class="paper-abstract">${p.abstract_snippet || p.abstract || "No abstract snippet available."}</p>
          </div>

          <div class="card-actions">
            <button class="btn-action btn-accent" onclick="app.openPaperModal(${p.id})">
              <span>Read Abstract</span> 📖
            </button>
            <button class="btn-action" onclick="app.findSimilarPapers(${p.id})">
              <span>Similar</span> 🔬
            </button>
            <button class="btn-icon-square ${isSaved ? 'bookmarked' : ''}" onclick="app.toggleSavePaper(${p.id})" title="${isSaved ? 'Remove from Saved' : 'Save Paper'}">
              ★
            </button>
          </div>
        </div>
      `;
    }).join("");
  }

  renderPapersPagination(page, totalPages) {
    const container = document.getElementById("papers-pagination");
    if (totalPages <= 1) {
      container.innerHTML = "";
      return;
    }
    container.innerHTML = `
      <button class="page-btn" ${page <= 1 ? "disabled" : ""} onclick="app.changePaperPage(${page - 1})">← Prev</button>
      <span class="page-info">Page ${page} of ${totalPages.toLocaleString()}</span>
      <button class="page-btn" ${page >= totalPages ? "disabled" : ""} onclick="app.changePaperPage(${page + 1})">Next →</button>
    `;
  }

  changePaperPage(newPage) {
    this.paperPage = newPage;
    this.loadPapers();
  }

  onPaperSearch() {
    clearTimeout(this.debounceTimer);
    this.debounceTimer = setTimeout(() => {
      this.paperQuery = document.getElementById("paper-search-input").value;
      this.paperPage = 1;
      this.loadPapers();
    }, 250);
  }

  onPaperCategoryFilter() {
    this.paperCategory = document.getElementById("paper-category-filter").value;
    this.paperPage = 1;
    this.loadPapers();
  }

  // Find Semantically Similar Papers to a Given Paper
  async findSimilarPapers(paperId) {
    this.switchTab("explorer");
    const alertBox = document.getElementById("paper-similarity-alert");
    alertBox.style.display = "block";
    alertBox.innerHTML = `
      <div style="background: #ffffff; border: 1px solid var(--border-color); border-radius: var(--radius-lg); padding: 1.25rem 1.5rem; display: flex; align-items: center; gap: 1rem; box-shadow: var(--shadow-sm);">
        <div class="stat-dot"></div>
        <span>Searching 10,000 papers for semantic neighbors of Paper #${paperId}...</span>
      </div>
    `;

    try {
      const res = await fetch(`${API_BASE}/api/papers/${paperId}/similar?limit=8`);
      if (!res.ok) throw new Error("Similar papers search failed");
      const data = await res.json();

      alertBox.innerHTML = `
        <div style="background: #ffffff; border: 1px solid var(--primary-border); border-radius: var(--radius-lg); padding: 1.25rem 1.5rem; display: flex; align-items: center; justify-content: space-between; box-shadow: var(--shadow-sm);">
          <div>
            <span style="font-size: 0.78rem; font-weight: 700; color: var(--primary); text-transform: uppercase;">Similar Papers Found for</span>
            <h4 style="font-size: 1.05rem; font-weight: 800; color: var(--text-main); margin-top: 0.15rem;">"${data.target_paper.title}"</h4>
          </div>
          <button class="btn-secondary" style="padding: 0.35rem 0.75rem; font-size: 0.8rem;" onclick="app.loadPapers(); document.getElementById('paper-similarity-alert').style.display='none';">
            Reset View ✕
          </button>
        </div>
      `;

      const grid = document.getElementById("papers-grid");
      this.renderPaperCards(data.similar_papers, grid, true);
      document.getElementById("papers-pagination").innerHTML = "";
    } catch (err) {
      console.error(err);
      alertBox.innerHTML = `<div style="color: red;">Failed to retrieve similar papers.</div>`;
    }
  }

  // Topic & Idea Matcher
  fillTopicPrompt(text) {
    document.getElementById("topic-input").value = text;
    this.executeTopicSearch();
  }

  async executeTopicSearch() {
    const input = document.getElementById("topic-input");
    const query = input.value.trim();
    if (!query) {
      this.showToast("Please enter a research topic or abstract idea.", "⚠️");
      return;
    }

    const container = document.getElementById("topic-results-container");
    container.style.display = "block";

    const rList = document.getElementById("topic-researchers-list");
    const pList = document.getElementById("topic-papers-list");

    rList.innerHTML = `<div class="skeleton-card"></div>`;
    pList.innerHTML = `<div class="skeleton-card"></div>`;

    document.getElementById("topic-researchers-count").textContent = "Matching...";
    document.getElementById("topic-papers-count").textContent = "Matching...";

    try {
      const res = await fetch(`${API_BASE}/api/search/topic`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, limit: 6 })
      });
      if (!res.ok) throw new Error("Topic search failed");
      const data = await res.json();

      document.getElementById("topic-researchers-count").textContent = `${data.researchers.length} Found`;
      document.getElementById("topic-papers-count").textContent = `${data.papers.length} Found`;

      // Render researchers list
      rList.innerHTML = data.researchers.map(r => `
        <div class="researcher-card" style="box-shadow: var(--shadow-sm);">
          <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem;">
            <div style="display: flex; gap: 0.75rem; align-items: center;">
              <div class="researcher-avatar" style="background: ${getAvatarGradient(r.researcher_name)}; width: 42px; height: 42px; font-size: 1rem;">
                ${getInitials(r.researcher_name)}
              </div>
              <div>
                <h4 class="researcher-name" style="font-size: 1.05rem;">${r.researcher_name}</h4>
                <div class="researcher-stats-row"><span>📄 ${r.paper_count} Paper</span></div>
              </div>
            </div>
            <span class="match-pill match-high">🔥 ${r.match_percent}% Fit</span>
          </div>
          <p class="research-snippet" style="-webkit-line-clamp: 2; margin-bottom: 0.75rem;">${r.research_snippet}</p>
          <div class="card-actions">
            <button class="btn-action btn-accent" onclick="app.matchCollaborators('${encodeURIComponent(r.researcher_name)}')">
              <span>Collaborate</span> 👥
            </button>
            <button class="btn-action" onclick="app.quickRecommendPapers('${encodeURIComponent(r.researcher_name)}')">
              <span>View Papers</span>
            </button>
          </div>
        </div>
      `).join("");

      // Render papers list
      pList.innerHTML = data.papers.map(p => `
        <div class="paper-card" style="box-shadow: var(--shadow-sm);">
          <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.35rem;">
            <span class="paper-date">${p.published || "Recent"}</span>
            <span class="match-pill match-medium">📑 ${p.match_percent}% Relevance</span>
          </div>
          <a href="${p.paper_id || '#'}" target="_blank" rel="noopener noreferrer" class="paper-title" style="font-size: 0.98rem;">
            ${p.title}
          </a>
          <p class="paper-abstract" style="-webkit-line-clamp: 2; margin-bottom: 0.75rem;">${p.abstract_snippet || p.abstract}</p>
          <div class="card-actions">
            <button class="btn-action btn-accent" onclick="app.openPaperModal(${p.id})">
              <span>Read Abstract</span>
            </button>
            <button class="btn-action" onclick="app.findSimilarPapers(${p.id})">
              <span>Similar</span>
            </button>
          </div>
        </div>
      `).join("");

      this.showToast("Synthesized semantic matches successfully!", "✨");
    } catch (err) {
      console.error(err);
      this.showToast("Failed to run topic match.", "❌");
    }
  }

  // Modals
  openPaperModal(paperId) {
    const paper = this.paperCache.get(paperId);
    if (!paper) return;

    const modal = document.getElementById("detail-modal");
    const body = document.getElementById("modal-body");

    const categoriesHtml = (paper.category_list || []).map(cat => 
      `<span class="badge-category ${getCategoryClass(cat)}">${cat}</span>`
    ).join(" ");

    body.innerHTML = `
      <div style="margin-bottom: 1rem;">
        <span style="font-size: 0.8rem; font-weight: 700; color: var(--text-subtle); text-transform: uppercase;">arXiv Paper #${paper.id} • ${paper.published || "Published"}</span>
        <h2 style="font-size: 1.35rem; font-weight: 800; color: var(--text-main); margin: 0.5rem 0 0.85rem; line-height: 1.35;">${paper.title}</h2>
        <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1.25rem;">
          ${categoriesHtml}
        </div>
      </div>

      <div style="background: var(--bg-main); border: 1px solid var(--border-color); border-radius: var(--radius-lg); padding: 1.25rem; font-size: 0.92rem; line-height: 1.65; color: var(--text-muted); max-height: 320px; overflow-y: auto; margin-bottom: 1.5rem;">
        <strong style="color: var(--text-main); display: block; margin-bottom: 0.5rem;">Abstract:</strong>
        ${paper.abstract || "No abstract text available for this publication."}
      </div>

      <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.75rem;">
        <a href="${paper.paper_id}" target="_blank" rel="noopener noreferrer" class="btn-primary">
          <span>Open on arXiv.org</span> ↗
        </a>
        <div style="display: flex; gap: 0.5rem;">
          <button class="btn-secondary" onclick="app.copyCitation('${paper.title.replace(/'/g, "\\'")}', '${paper.paper_id}')">
            <span>Copy Citation</span> 📋
          </button>
          <button class="btn-secondary" onclick="app.findSimilarPapers(${paper.id}); app.closeModal();">
            <span>Find Similar Papers</span> 🔬
          </button>
        </div>
      </div>
    `;

    modal.classList.add("open");
  }

  closeModal(event) {
    if (event && event.target !== event.currentTarget && !event.target.classList.contains("modal-close-btn")) {
      return;
    }
    document.getElementById("detail-modal").classList.remove("open");
  }

  copyCitation(title, url) {
    const citation = `"${title}". arXiv. Available at: ${url}`;
    navigator.clipboard.writeText(citation).then(() => {
      this.showToast("Citation copied to clipboard!", "📋");
    });
  }

  // Saved / Bookmarks Management
  toggleSaveResearcher(id) {
    const r = this.researcherCache.get(id);
    if (!r) return;

    const idx = this.savedResearchers.findIndex(x => x.id === id);
    if (idx >= 0) {
      this.savedResearchers.splice(idx, 1);
      this.showToast(`Removed ${r.researcher_name} from Saved`, "🗑️");
    } else {
      this.savedResearchers.push({
        id: r.id,
        name: r.researcher_name,
        categories: r.category_list || [],
        paper_count: r.paper_count || 1
      });
      this.showToast(`Saved ${r.researcher_name} to Shortlist!`, "⭐");
    }

    localStorage.setItem("saved_researchers", JSON.stringify(this.savedResearchers));
    this.updateSavedBadge();
    this.loadResearchers();
    if (this.currentTab === "saved") this.renderSaved();
  }

  toggleSavePaper(id) {
    const p = this.paperCache.get(id);
    if (!p) return;

    const idx = this.savedPapers.findIndex(x => x.id === id);
    if (idx >= 0) {
      this.savedPapers.splice(idx, 1);
      this.showToast("Removed paper from Saved", "🗑️");
    } else {
      this.savedPapers.push({
        id: p.id,
        title: p.title,
        paper_id: p.paper_id,
        published: p.published,
        categories: p.category_list || []
      });
      this.showToast("Saved paper to Shortlist!", "⭐");
    }

    localStorage.setItem("saved_papers", JSON.stringify(this.savedPapers));
    this.updateSavedBadge();
    this.loadPapers();
    if (this.currentTab === "saved") this.renderSaved();
  }

  updateSavedBadge() {
    const count = this.savedResearchers.length + this.savedPapers.length;
    document.getElementById("saved-counter").textContent = count;
  }

  renderSaved() {
    const rContainer = document.getElementById("saved-researchers-list");
    const pContainer = document.getElementById("saved-papers-list");

    document.getElementById("saved-researchers-count").textContent = `${this.savedResearchers.length} saved`;
    document.getElementById("saved-papers-count").textContent = `${this.savedPapers.length} saved`;

    if (!this.savedResearchers.length) {
      rContainer.innerHTML = `<p style="color: var(--text-subtle); padding: 1rem; background: #ffffff; border-radius: var(--radius-md); border: 1px dashed var(--border-color);">No researchers bookmarked yet. Click the ★ icon on any researcher card.</p>`;
    } else {
      rContainer.innerHTML = this.savedResearchers.map(r => `
        <div class="researcher-card" style="padding: 1rem;">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; gap: 0.75rem; align-items: center;">
              <div class="researcher-avatar" style="background: ${getAvatarGradient(r.name)}; width: 36px; height: 36px; font-size: 0.85rem;">
                ${getInitials(r.name)}
              </div>
              <div>
                <h4 style="font-size: 0.95rem; font-weight: 700; color: var(--text-main);">${r.name}</h4>
                <span style="font-size: 0.75rem; color: var(--text-subtle);">${r.paper_count} Paper • ID #${r.id}</span>
              </div>
            </div>
            <div style="display: flex; gap: 0.4rem;">
              <button class="btn-secondary" style="padding: 0.35rem 0.65rem; font-size: 0.78rem;" onclick="app.matchCollaborators('${encodeURIComponent(r.name)}')">Match</button>
              <button class="btn-icon-square bookmarked" style="width: 28px; height: 28px; font-size: 0.8rem;" onclick="app.toggleSaveResearcher(${r.id})">★</button>
            </div>
          </div>
        </div>
      `).join("");
    }

    if (!this.savedPapers.length) {
      pContainer.innerHTML = `<p style="color: var(--text-subtle); padding: 1rem; background: #ffffff; border-radius: var(--radius-md); border: 1px dashed var(--border-color);">No papers bookmarked yet. Click the ★ icon on any paper card.</p>`;
    } else {
      pContainer.innerHTML = this.savedPapers.map(p => `
        <div class="paper-card" style="padding: 1rem;">
          <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 0.5rem; margin-bottom: 0.5rem;">
            <a href="${p.paper_id}" target="_blank" class="paper-title" style="font-size: 0.92rem; margin-bottom: 0;">${p.title}</a>
            <button class="btn-icon-square bookmarked" style="width: 28px; height: 28px; font-size: 0.8rem; flex-shrink: 0;" onclick="app.toggleSavePaper(${p.id})">★</button>
          </div>
          <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.75rem; color: var(--text-subtle);">
            <span>📅 ${p.published || "Recent"}</span>
            <button class="btn-action" style="padding: 0.25rem 0.5rem; font-size: 0.75rem;" onclick="app.findSimilarPapers(${p.id})">Find Similar</button>
          </div>
        </div>
      `).join("");
    }
  }

  exportShortlist() {
    if (!this.savedResearchers.length && !this.savedPapers.length) {
      this.showToast("Shortlist is currently empty!", "ℹ️");
      return;
    }
    const data = {
      exported_at: new Date().toISOString(),
      shortlisted_researchers: this.savedResearchers,
      shortlisted_papers: this.savedPapers
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `research_collaboration_shortlist_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    this.showToast("Shortlist exported successfully!", "📥");
  }

  clearSaved() {
    if (!confirm("Are you sure you want to clear your saved shortlist?")) return;
    this.savedResearchers = [];
    this.savedPapers = [];
    localStorage.removeItem("saved_researchers");
    localStorage.removeItem("saved_papers");
    this.updateSavedBadge();
    this.renderSaved();
    this.loadResearchers();
    this.loadPapers();
    this.showToast("Shortlist cleared.", "🗑️");
  }
}

// Global instance
const app = new App();
window.addEventListener("DOMContentLoaded", () => {
  app.init();
});
