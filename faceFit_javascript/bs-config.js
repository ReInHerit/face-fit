module.exports = {
    proxy: `${process.env.HOST || 'localhost'}:${process.env.PORT || 8000}`,
    files: ["**/*.css", "**/*.pug", "**/*.js"],
    ignore: ["node_modules"],
    reloadDelay: 10,
    ui: false,
    notify: false,
    port: 3000,
};